import os, json, argparse
from typing import List, Dict, Tuple, Iterator
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, Sampler

from transformers import BertModel, BertTokenizerFast, get_linear_schedule_with_warmup


def _read_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


@dataclass
class Row:
    text: str
    cui: str
    label: int


class BiJsonLDataset(Dataset):
    def __init__(self, jsonl_path: str, cui_to_id: Dict[str, int]):
        rows = _read_jsonl(jsonl_path)
        self.items: List[Row] = []
        for r in rows:
            cui = str(r.get("cui", ""))
            txt = str(r.get("text", "")).strip()
            if not txt or cui not in cui_to_id:
                continue
            self.items.append(Row(text=txt, cui=cui, label=int(cui_to_id[cui])))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Row:
        return self.items[idx]


class BalancedBatchSampler(Sampler[List[int]]):
    def __init__(self, dataset: BiJsonLDataset, samples_per_cui: int, cuis_per_batch: int, seed: int = 13, max_per_cui: int = 8):
        self.ds = dataset
        self.samples_per_cui = int(samples_per_cui)
        self.cuis_per_batch = int(cuis_per_batch)
        self.max_per_cui = int(max_per_cui)
        self.rng = torch.Generator()
        self.rng.manual_seed(seed)

        by_label: Dict[int, List[int]] = {}
        for i, row in enumerate(self.ds.items):
            by_label.setdefault(row.label, []).append(i)
        self.by_label = by_label
        self.labels = list(by_label.keys())

        # Estimate number of batches per epoch; when samples_per_cui <= 0 ("all" mode),
        # approximate using max_per_cui to avoid exploding total steps
        denom_spc = self.samples_per_cui if self.samples_per_cui > 0 else max(1, self.max_per_cui)
        self._len = max(1, len(self.ds) // max(1, denom_spc * self.cuis_per_batch))

    def __len__(self) -> int:
        return self._len

    def __iter__(self) -> Iterator[List[int]]:
        labels_tensor = torch.tensor(self.labels, dtype=torch.long)
        for _ in range(self._len):
            if len(self.labels) <= self.cuis_per_batch:
                chosen = labels_tensor
            else:
                idx = torch.randperm(len(self.labels), generator=self.rng)[:self.cuis_per_batch]
                chosen = labels_tensor[idx]
            batch_indices: List[int] = []
            for lab in chosen.tolist():
                pool = self.by_label[lab]
                # Determine how many samples to take for this CUI
                if self.samples_per_cui <= 0:
                    take_n = min(len(pool), self.max_per_cui)
                else:
                    take_n = self.samples_per_cui

                if len(pool) >= take_n:
                    sel_idx = torch.randperm(len(pool), generator=self.rng)[:take_n].tolist()
                    batch_indices.extend([pool[k] for k in sel_idx])
                else:
                    # Only occurs when samples_per_cui > 0 and CUI has fewer items; fill by resampling
                    reps = [pool[int(torch.randint(0, len(pool), (1,), generator=self.rng))] for _ in range(take_n)]
                    batch_indices.extend(reps)
            perm = torch.randperm(len(batch_indices), generator=self.rng).tolist()
            yield [batch_indices[i] for i in perm]


def masked_mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


def supcon_loss(emb: torch.Tensor, labels: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
    sim = (emb @ emb.t()) / float(tau)
    B = emb.size(0)
    eye = torch.eye(B, device=emb.device, dtype=emb.dtype)
    pos_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float() * (1.0 - eye)
    logits = sim - 1e9 * eye
    exp_logits = torch.exp(logits)
    denom = exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12)
    pos_sum = (exp_logits * pos_mask).sum(dim=1, keepdim=True).clamp_min(1e-12)
    pos_count = pos_mask.sum(dim=1).clamp_min(1.0)
    loss_i = -torch.log(pos_sum / denom).squeeze(1) / pos_count
    return loss_i.mean()


@torch.no_grad()
def encode_texts(model: BertModel, tokenizer: BertTokenizerFast, texts: List[str], device: torch.device,
                 batch_size: int = 512, max_length: int = 64) -> torch.Tensor:
    vecs = []
    model.eval()
    for start in range(0, len(texts), batch_size):
        chunk = texts[start:start + batch_size]
        toks = tokenizer(chunk, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        toks = {k: v.to(device) for k, v in toks.items()}
        out = model(**toks).last_hidden_state
        pooled = masked_mean_pool(out, toks["attention_mask"])
        vecs.append(F.normalize(pooled, p=2, dim=1))
    return torch.cat(vecs, dim=0) if vecs else torch.empty((0, model.config.hidden_size), device=device)


def evaluate_alias_recall(dev_ds: BiJsonLDataset,
                          all_cands: List[Tuple[str, int]],
                          model: BertModel,
                          tokenizer: BertTokenizerFast,
                          device: torch.device,
                          k: int = 10,
                          max_cands: int = -1,
                          max_queries: int = -1) -> Tuple[float, float, int]:
    cand_texts = [t for t, _ in all_cands]
    cand_labels = [lab for _, lab in all_cands]
    if max_cands is not None and max_cands > 0 and len(cand_texts) > max_cands:
        cand_texts = cand_texts[:max_cands]
        cand_labels = cand_labels[:max_cands]

    q_texts = [r.text for r in dev_ds.items]
    q_labels = [r.label for r in dev_ds.items]
    if max_queries is not None and max_queries > 0 and len(q_texts) > max_queries:
        q_texts = q_texts[:max_queries]
        q_labels = q_labels[:max_queries]

    if not q_texts or not cand_texts:
        return 0.0, 0.0, 0

    cand_mat = encode_texts(model, tokenizer, cand_texts, device, batch_size=1024, max_length=64)
    q_mat = encode_texts(model, tokenizer, q_texts, device, batch_size=1024, max_length=64)

    sims = q_mat @ cand_mat.t()
    Nq, Nc = sims.size()
    Kprime = min(max(50, k * 5), Nc)
    topk_idx = torch.topk(sims, k=Kprime, dim=1).indices

    correct_at1 = 0
    correct_atK = 0
    for i in range(Nq):
        seen = set()
        unique_labels: List[int] = []
        for pos in topk_idx[i].tolist():
            lab = int(cand_labels[pos])
            if lab not in seen:
                seen.add(lab)
                unique_labels.append(lab)
            if len(unique_labels) == k:
                break
        if unique_labels:
            if unique_labels[0] == q_labels[i]:
                correct_at1 += 1
            if q_labels[i] in unique_labels:
                correct_atK += 1
    return correct_at1 / Nq, correct_atK / Nq, Nq


def main():
    ap = argparse.ArgumentParser(description="Fine-tune SapBERT bi-encoder with supervised contrastive loss.")
    # Data
    ap.add_argument("--data_dir", type=str, default="improved_less_synonyms/processed/bi_contrastive")
    ap.add_argument("--train_file", type=str, default="train.jsonl")
    ap.add_argument("--dev_file", type=str, default="dev.jsonl", help="Optional dev file; if missing, dev is skipped")
    # Model/optim
    ap.add_argument("--encoder_name", type=str, default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    ap.add_argument("--hf_cache_dir", type=str, default=None, help="Optional HF cache dir (overrides env)")
    ap.add_argument("--max_length", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--amp", action="store_true")
    # Batching
    ap.add_argument("--cuis_per_batch", type=int, default=64)
    ap.add_argument("--samples_per_cui", type=int, default=2)
    ap.add_argument("--max_per_cui", type=int, default=8, help="Cap for per-CUI samples when samples_per_cui<=0")
    ap.add_argument("--grad_accum_steps", type=int, default=1)
    # Eval
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--eval_k", type=int, default=10)
    ap.add_argument("--eval_max_cands", type=int, default=-1)
    ap.add_argument("--eval_max_queries", type=int, default=-1)
    # Save
    ap.add_argument("--save_dir", type=str, default="saved_models/bi_encoder_contrastive")
    args = ap.parse_args()
    # Disable dev evaluation entirely
    DO_EVAL = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Honor HF cache dir similar to CE script if provided or via env HF_HOME
    cache_dir = None
    if args.hf_cache_dir:
        cache_dir = args.hf_cache_dir
    else:
        cache_dir = os.environ.get("HF_HOME") or None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        # Common envs respected by HF Hub/Transformers
        os.environ["HF_HOME"] = cache_dir
        os.environ["TRANSFORMERS_CACHE"] = cache_dir
        os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir

    # Robust loading: respect cache_dir and avoid safetensors issues if present
    def _load_tokenizer(name: str):
        try:
            return BertTokenizerFast.from_pretrained(name, cache_dir=cache_dir)
        except Exception:
            # fallback without cache_dir env
            return BertTokenizerFast.from_pretrained(name)

    def _load_model(name: str) -> BertModel:
        # Try normal load first (cache_dir used if set)
        try:
            return BertModel.from_pretrained(name, cache_dir=cache_dir).to(device)
        except Exception:
            # Retry disabling safetensors usage
            try:
                return BertModel.from_pretrained(name, cache_dir=cache_dir, use_safetensors=False).to(device)
            except Exception:
                # Last fallback: no cache_dir hint
                return BertModel.from_pretrained(name, use_safetensors=False).to(device)

    tokenizer = _load_tokenizer(args.encoder_name)
    model = _load_model(args.encoder_name)

    train_path = os.path.join(args.data_dir, args.train_file)
    dev_path = os.path.join(args.data_dir, args.dev_file) if args.dev_file else None
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing train: {train_path}")

    # Build CUI id map from TRAIN ONLY
    train_rows = _read_jsonl(train_path)
    cui_list = sorted(list({str(r["cui"]) for r in train_rows}))
    cui_to_id = {c: i for i, c in enumerate(cui_list)}

    train_ds = BiJsonLDataset(train_path, cui_to_id)
    dev_exists = bool(dev_path) and os.path.exists(dev_path)
    dev_ds = BiJsonLDataset(dev_path, cui_to_id) if dev_exists else None

    # Candidates for optional eval: from TRAIN ONLY
    all_cands: List[Tuple[str, int]] = [(r["text"], cui_to_id[str(r["cui"])]) for r in train_rows]

    sampler = BalancedBatchSampler(train_ds, samples_per_cui=int(args.samples_per_cui),
                                   cuis_per_batch=int(args.cuis_per_batch), seed=13,
                                   max_per_cui=int(args.max_per_cui))

    def collate(batch_rows: List[Row]):
        texts = [r.text for r in batch_rows]
        labels = torch.tensor([r.label for r in batch_rows], dtype=torch.long)
        toks = tokenizer(texts, padding=True, truncation=True, max_length=int(args.max_length), return_tensors="pt")
        return toks, labels

    loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=2, pin_memory=(device.type == "cuda"),
                        collate_fn=collate)

    total_steps = len(loader) * int(args.epochs)
    warmup_steps = int(args.warmup_ratio * total_steps)
    optim = AdamW(model.parameters(), lr=float(args.lr))
    sched = get_linear_schedule_with_warmup(optim, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp) and device.type == "cuda")

    os.makedirs(args.save_dir, exist_ok=True)
    best_r1 = -1.0
    step = 0

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        for toks, labels in loader:
            step += 1
            toks = {k: v.to(device, non_blocking=True) for k, v in toks.items()}
            labels = labels.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=bool(args.amp) and device.type == "cuda"):
                out = model(**toks).last_hidden_state
                pooled = masked_mean_pool(out, toks["attention_mask"])
                emb = F.normalize(pooled, p=2, dim=1)
                loss = supcon_loss(emb, labels, tau=0.07)

            scaler.scale(loss).backward()
            if step % int(args.grad_accum_steps) == 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                sched.step()

            if step % 100 == 0:
                print(f"[epoch {epoch}] step {step}/{total_steps} loss={float(loss.item()):.4f}")

            if DO_EVAL and dev_ds is not None and args.eval_every > 0 and step % int(args.eval_every) == 0:
                r1, rK, N = evaluate_alias_recall(
                    dev_ds, all_cands, model, tokenizer, device,
                    k=int(args.eval_k),
                    max_cands=int(args.eval_max_cands),
                    max_queries=int(args.eval_max_queries),
                )
                print(f"[eval] recall@1={r1:.4f} recall@{args.eval_k}={rK:.4f} (N={N})")
                if r1 >= best_r1:
                    best_r1 = r1
                    save_path = os.path.join(args.save_dir, "sapbert_supcon_best")
                    model.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)
                    with open(os.path.join(save_path, "meta.json"), "w") as f:
                        json.dump({"recall_at_1": float(r1), "recall_at_k": float(rK), "k": int(args.eval_k), "N": int(N)}, f, indent=2)
                    print(f"Saved checkpoint: {save_path}")

        if DO_EVAL and dev_ds is not None:
            r1, rK, N = evaluate_alias_recall(
                dev_ds, all_cands, model, tokenizer, device,
                k=int(args.eval_k),
                max_cands=int(args.eval_max_cands),
                max_queries=int(args.eval_max_queries),
            )
            print(f"[epoch {epoch} done] recall@1={r1:.4f} recall@{args.eval_k}={rK:.4f} (N={N})")
            if r1 >= best_r1:
                best_r1 = r1
                save_path = os.path.join(args.save_dir, "sapbert_supcon_best")
                model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                with open(os.path.join(save_path, "meta.json"), "w") as f:
                    json.dump({"recall_at_1": float(r1), "recall_at_k": float(rK), "k": int(args.eval_k), "N": int(N)}, f, indent=2)
                print(f"Saved checkpoint: {save_path}")
        else:
            # Save checkpoint at end of each epoch without evaluation
            save_path = os.path.join(args.save_dir, "sapbert_supcon_last")
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"Saved checkpoint: {save_path}")


if __name__ == "__main__":
    main()


