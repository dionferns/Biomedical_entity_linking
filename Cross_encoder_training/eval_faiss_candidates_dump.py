import os
import json
import argparse
from typing import List, Tuple, Dict

import numpy as np
import torch
import faiss
from transformers import BertModel, BertTokenizerFast
import gzip


def load_tokenized_ids(tokenized_path: str) -> List[int]:
    if not os.path.exists(tokenized_path):
        raise FileNotFoundError(f"Tokenized file not found: {tokenized_path}")
    data = torch.load(tokenized_path, map_location='cpu')
    mention_ids: List[int] = []
    seen = set()
    for ex_list in data.values():
        for ex in ex_list:
            labels = ex["labels"].tolist()
            ids = ex["id_tensor"].tolist()
            for lab, mid in zip(labels, ids):
                if lab == 1 and mid > 0 and mid not in seen:
                    seen.add(mid)
                    mention_ids.append(int(mid))
    return mention_ids


def load_id2cuistr(id2_path: str) -> dict:
    if not os.path.exists(id2_path):
        raise FileNotFoundError(f"id2cuistr file not found: {id2_path}")
    with open(id2_path, 'r', encoding='utf-8') as f:
        m = json.load(f)
    return {str(k): v for k, v in m.items()}


def encode_texts(texts: List[str], encoder_name: str, device: torch.device, batch_size: int = 64, amp: bool = False, auto_batch: bool = False) -> np.ndarray:
    if not texts:
        raise ValueError("No texts provided for encoding.")
    tokenizer = BertTokenizerFast.from_pretrained(encoder_name)
    encoder = BertModel.from_pretrained(encoder_name).to(device)
    encoder.eval()
    # Diagnostics: encoder training state and a tiny checksum to verify weights
    try:
        first_param = next(encoder.parameters())
        checksum = float(first_param.detach().abs().view(-1)[:1000].sum().item())
        print(f"[diag] query encoder.training={encoder.training} | checksum={checksum:.6e}")
    except Exception:
        pass

    def _run_encode(cur_bs: int) -> np.ndarray:
        all_vecs: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(texts), cur_bs):
                chunk = texts[start:start + cur_bs]
                toks = tokenizer(
                    chunk,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors='pt',
                ).to(device)
                if amp:
                    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=(device.type == 'cuda')):
                        outputs = encoder(**toks).last_hidden_state
                else:
                    outputs = encoder(**toks).last_hidden_state  # [B, T, H]
                mask = toks["attention_mask"].unsqueeze(-1).float()  # [B, T, 1]
                masked_sum = (outputs * mask).sum(dim=1)  # [B, H]
                lengths = mask.sum(dim=1).clamp(min=1.0)  # [B, 1]
                vecs = (masked_sum / lengths).detach().cpu().numpy().astype('float32')
                all_vecs.append(vecs)
        queries = np.concatenate(all_vecs, axis=0) if all_vecs else np.zeros((0, 768), dtype='float32')
        norms = np.linalg.norm(queries, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        queries = queries / norms
        return queries.astype('float32', copy=False)

    if not auto_batch:
        try:
            return _run_encode(batch_size)
        except torch.cuda.OutOfMemoryError as e:
            raise RuntimeError(f"OOM during encoding with batch_size={batch_size}. Reduce --batch_size or enable --auto_batch. Original error: {e}")
    else:
        if device.type != 'cuda':
            raise RuntimeError("--auto_batch requires CUDA. No GPU available.")
        # Simple adaptive probing to maximize usable batch size without OOM.
        bs = max(1, int(batch_size))
        best_ok = 0
        last_err = None
        while True:
            try:
                torch.cuda.empty_cache()
                out = _run_encode(bs)
                best_ok = bs
                # Try to grow; cap to len(texts)
                if bs >= len(texts):
                    return out
                grow = min(len(texts), bs * 2)
                if grow == bs:
                    return out
                bs = grow
                # If we can encode entire dataset in one go, return
                if bs >= len(texts):
                    return _run_encode(bs)
            except torch.cuda.OutOfMemoryError as e:
                last_err = e
                torch.cuda.empty_cache()
                if best_ok == 0:
                    raise RuntimeError(f"OOM during adaptive batching at batch_size={bs}. Try a smaller --batch_size. Original error: {e}")
                # Back off by half
                new_bs = max(1, bs // 2)
                if new_bs == bs:
                    # Cannot reduce further
                    raise RuntimeError(f"OOM and cannot reduce batch size further (bs={bs}). Original error: {e}")
                bs = new_bs


def load_faiss_index_and_cuids(index_path: str = None, cuids_path: str = None) -> Tuple[faiss.Index, List[str]]:
    if index_path is None or cuids_path is None:
        # base_dir = os.path.join('improvedd', 'processed', 'entity_disambiguation')
        # index_path = index_path or os.path.join(base_dir, 'faiss_index_aug2_name_full.bin')
        # cuids_path = cuids_path or os.path.join(base_dir, 'cuids_aug2_name_full.json')

        # base_dir = os.path.join('data', 'processed', 'entity_disambiguation')
        # index_path = index_path or os.path.join(base_dir, 'faiss_index_aug_name_full.bin')
        # cuids_path = cuids_path or os.path.join(base_dir, 'cuids_aug_name_full.json')

        base_dir = os.path.join('improved_less_synonyms', 'processed', 'entity_disambiguation')
        index_path = index_path or os.path.join(base_dir, 'faiss_index_aug2_name_full.bin')
        cuids_path = cuids_path or os.path.join(base_dir, 'cuids_aug2_name_full.json')

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    if not os.path.exists(cuids_path):
        raise FileNotFoundError(f"CUIDs file not found: {cuids_path}")
    index = faiss.read_index(index_path)
    with open(cuids_path, 'r', encoding='utf-8') as f:
        cuids = json.load(f)
    return index, cuids


def compute_accuracy(golds: List[str], preds_topk: List[List[str]], k: int) -> float:
    if not golds:
        return 0.0
    correct = 0
    for gold, pred_list in zip(golds, preds_topk):
        topk = pred_list[:k]
        if gold in topk:
            correct += 1
    return correct / len(golds)


def dedup_topk_by_cui(idx_row: np.ndarray, cuids: List[str], k: int) -> List[str]:
    unique: List[str] = []
    seen = set()
    for pos in idx_row:
        if pos < 0 or pos >= len(cuids):
            continue
        cui = cuids[pos]
        if cui in seen:
            continue
        seen.add(cui)
        unique.append(cui)
        if len(unique) == k:
            break
    return unique


def topk_unique_pos_by_cui(idx_row: np.ndarray, cuids: List[str], k: int) -> List[int]:
    unique_pos: List[int] = []
    seen = set()
    for pos in idx_row:
        if pos < 0 or pos >= len(cuids):
            continue
        cui = cuids[pos]
        if cui in seen:
            continue
        seen.add(cui)
        unique_pos.append(int(pos))
        if len(unique_pos) == k:
            break
    return unique_pos


def build_candidates_dict(
    kept_ids: List[str],
    texts: List[str],
    golds: List[str],
    idx: np.ndarray,
    cuids: List[str],
    names_by_pos: List[str],
    K: int,
) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for ex_i, mention_id in enumerate(kept_ids):
        pos_list = topk_unique_pos_by_cui(idx[ex_i], cuids, K)
        gold_cui = golds[ex_i]
        candidates = []
        for rank, pos in enumerate(pos_list, 1):
            cand_cui = cuids[pos]
            candidates.append({
                "cui": cand_cui,
                "name": names_by_pos[pos],
                "is_gold": bool(cand_cui == gold_cui),
                "is_null": False,
                "faiss_rank": str(rank),
            })
        # Append a NIL candidate if the gold CUI was not retrieved
        if not any(c.get("is_gold", False) for c in candidates):
            candidates.append({
                "cui": "[NIL]",
                "name": "[NIL]",
                "is_gold": True,
                "is_null": True,
                "faiss_rank": str(K + 1),
            })
        out[str(mention_id)] = {
            "mention": texts[ex_i],
            "gold_cui": gold_cui,
            "candidates": candidates,
        }
    return out


def main():
    parser = argparse.ArgumentParser(description='Evaluate FAISS retrieval (synonyms index) with per-query CUI dedup to K unique results.')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'])
    parser.add_argument('--encoder_name', type=str, default='cambridgeltl/SapBERT-from-PubMedBERT-fulltext')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--overshoot_mult', type=float, default=5.0, help='Multiplier to compute K prime (K\' = max(50, overshoot_mult*K)).')
    parser.add_argument('--tokenized_path', type=str, default=None)
    parser.add_argument('--id2cuistr_path', type=str, default=None)
    parser.add_argument('--ent_def_path', type=str, default=None)
    parser.add_argument('--index_path', type=str, default=None)
    parser.add_argument('--cuids_path', type=str, default=None)
    parser.add_argument('--amp', action='store_true', help='Enable mixed precision for encoding (CUDA only).')
    parser.add_argument('--auto_batch', action='store_true', help='Enable adaptive batching to maximize GPU usage (CUDA only).')
    parser.add_argument('--use_faiss_gpu', action='store_true', help='Clone FAISS index to GPU for searching (CUDA only).')
    parser.add_argument('--search_batch_size', type=int, default=0, help='If >0, search queries in batches of this size to control memory.')
    parser.add_argument('--debug', action='store_true', help='Use debug MedMention inputs; FAISS index remains unchanged.')
    parser.add_argument('--candidates_json_out', type=str, default=None, help='Path to write candidate_data JSON. Defaults to cross_encoder/candidate_data_<split>.json')
    parser.add_argument('--faiss_fp16_index', action='store_true', help='Store FAISS vectors on GPU in float16 to save memory.')
    parser.add_argument('--faiss_temp_mem_mb', type=int, default=256, help='FAISS GPU temporary memory in MB (lower to avoid OOM).')
    args = parser.parse_args()

    if args.K <= 0:
        raise ValueError(f"K must be positive, got {args.K}")
    if args.overshoot_mult <= 1.0:
        raise ValueError(f"overshoot_mult must be > 1.0, got {args.overshoot_mult}")

    # Derive defaults for MD artifacts (unchanged locations)
    md_dir = os.path.join('data', 'processed', 'mention_detection', args.split)
    suffix = 'debug' if args.debug else 'full'
    if args.tokenized_path is None:
        args.tokenized_path = os.path.join(md_dir, f'tokenized_{args.split}_{suffix}.pt')
    if args.id2cuistr_path is None:
        args.id2cuistr_path = os.path.join(md_dir, f'id2cuistr_dict_{suffix}.json')

    # Use improved ent_def by default (stored with other artifacts under data/processed)
    # ent_dir = os.path.join('improvedd', 'processed', 'entity_disambiguation')
    # ent_dir = os.path.join('data', 'processed', 'entity_disambiguation')
    ent_dir = os.path.join('improved_less_synonyms', 'processed', 'entity_disambiguation')

    if args.ent_def_path is None:
        args.ent_def_path = os.path.join(ent_dir, 'ent_def_aug2_full_syn.json')

    # Load inputs (fail fast on missing files)
    print(f"Using tokenized: {args.tokenized_path}")
    print(f"Using id2cuistr: {args.id2cuistr_path}")
    print(f"Using ent_def: {args.ent_def_path}")
    print(f"Index mode: {'debug' if args.debug else 'full'} | Encoder: {args.encoder_name}")
    mention_ids = load_tokenized_ids(args.tokenized_path)
    id2 = load_id2cuistr(args.id2cuistr_path)
    if not os.path.exists(args.ent_def_path):
        raise FileNotFoundError(f"CUI-to-name mapping file not found: {args.ent_def_path}")
    with open(args.ent_def_path, 'r', encoding='utf-8') as f:
        ent_list = json.load(f)
    ent_def = {rec['cui']: rec for rec in ent_list}
    # Names aligned to FAISS positions: must match the order used to build the index
    names_by_pos = [rec.get('name', '') for rec in ent_list]
    if len(names_by_pos) == 0:
        raise RuntimeError("Entity list is empty; cannot format examples.")

    # Build mention texts and gold CUIs
    texts: List[str] = []
    golds: List[str] = []
    kept_ids: List[str] = []
    for mid in mention_ids:
        rec = id2.get(str(mid))
        if not rec:
            continue
        mention_text = (rec.get('mention') or '').strip()
        gold_cui = (rec.get('cui') or '').strip()
        if mention_text and gold_cui and gold_cui != 'UNK':
            texts.append(mention_text)
            golds.append(gold_cui)
            kept_ids.append(str(mid))

    if not texts:
        raise RuntimeError("No valid mentions found to evaluate.")
    # Diagnostics: mention stats and a few samples
    try:
        num_mentions = len(texts)
        num_blanks = sum(1 for t in texts if not t)
        print(f"[diag] mentions kept={num_mentions} | blanks_after_filter={num_blanks}")
        for i in range(min(3, num_mentions)):
            sample_txt = texts[i][:80]
            print(f"[sample {i}] text='{sample_txt}' gold='{golds[i]}'")
    except Exception:
        pass

    # Encode queries (optionally with AMP and adaptive batching)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.amp and device.type != 'cuda':
        raise RuntimeError("--amp requires CUDA but no GPU is available.")
    if args.auto_batch and device.type != 'cuda':
        raise RuntimeError("--auto_batch requires CUDA but no GPU is available.")
    queries = encode_texts(texts, args.encoder_name, device=device, batch_size=args.batch_size, amp=args.amp, auto_batch=args.auto_batch)
    # Diagnostics: query vector norms after L2-normalization
    try:
        q_norms = np.linalg.norm(queries, axis=1)
        if q_norms.size:
            print(f"[diag] query norms: min={q_norms.min():.4f} mean={q_norms.mean():.4f} max={q_norms.max():.4f}")
    except Exception:
        pass

    # Load FAISS index and CUIDs
    index, cuids = load_faiss_index_and_cuids(args.index_path, args.cuids_path)

    # Optionally move index to GPU
    if args.use_faiss_gpu:
        if not torch.cuda.is_available():
            raise RuntimeError("--use_faiss_gpu requested but CUDA is not available.")
        try:
            # Free any leftover CUDA memory from encoding
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

            res = faiss.StandardGpuResources()
            # Reduce FAISS preallocated temp memory (default ~1.5 GB)
            try:
                res.setTempMemory(int(args.faiss_temp_mem_mb) * 1024 * 1024)
            except Exception:
                pass

            co = faiss.GpuClonerOptions()
            # Optional: halve index memory usage
            if getattr(args, 'faiss_fp16_index', False):
                co.useFloat16 = True

            index = faiss.index_cpu_to_gpu(res, 0, index, co)
        except Exception as e1:
            # Retry with strictest memory settings
            try:
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                res = faiss.StandardGpuResources()
                try:
                    res.noTempMemory()
                except Exception:
                    pass
                co = faiss.GpuClonerOptions()
                co.useFloat16 = True
                index = faiss.index_cpu_to_gpu(res, 0, index, co)
                print("[warn] Retried FAISS GPU clone with noTempMemory + FP16.")
            except Exception as e2:
                raise RuntimeError(f"Failed to move FAISS index to GPU after retries: first={e1} | retry={e2}")
    # Index diagnostics
    try:
        idx_type = type(index).__name__
        ntotal = int(getattr(index, 'ntotal', 0))
        print(f"[diag] index type={idx_type} | ntotal={ntotal}")
    except Exception:
        pass

    # Compute overshoot K'
    K = int(args.K)
    Kprime = max(50, int(args.overshoot_mult * K))
    # For very small indices, enforce feasibility
    if hasattr(index, 'ntotal') and index.ntotal is not None:
        Kprime = min(Kprime, int(index.ntotal))
        if Kprime < K:
            raise RuntimeError(f"Index too small for requested K. ntotal={index.ntotal}, requested K={K}")

    # Search
    if args.search_batch_size and args.search_batch_size > 0:
        scores, idx = search_in_chunks(index, queries, Kprime, args.search_batch_size)
    else:
        scores, idx = index.search(queries, Kprime)

    # Diagnostic: per-query unique CUI count within K'
    diag_mean_unique = None
    diag_median_unique = None
    diag_pct_lt_k = None
    try:
        unique_counts: List[int] = []
        insufficient = 0
        for row in idx:
            seen = set()
            for pos in row:
                if 0 <= pos < len(cuids):
                    seen.add(cuids[pos])
            c = len(seen)
            unique_counts.append(c)
            if c < K:
                insufficient += 1
        if unique_counts:
            diag_mean_unique = float(np.mean(unique_counts))
            diag_median_unique = int(np.median(unique_counts))
            diag_pct_lt_k = 100.0 * (insufficient / len(unique_counts))
            print(f"[diag] unique_CUIs_within_Kprime: mean={diag_mean_unique:.2f} median={diag_median_unique} pct_lt_K={diag_pct_lt_k:.2f}%")
    except Exception:
        pass

    # Sample triples: (mention_text, gold_cui, top1_cui)
    try:
        for i in range(min(3, len(texts))):
            top1_pos = idx[i][0] if idx is not None and len(idx[i]) > 0 else -1
            top1 = cuids[top1_pos] if (top1_pos >= 0 and top1_pos < len(cuids)) else None
            print(f"[triplet {i}] gold={golds[i]} top1={top1} text='{texts[i][:80]}'")
    except Exception:
        pass

    # Print 10 examples with top-10 unique candidates (name + CUI)
    if len(cuids) != len(names_by_pos):
        raise RuntimeError(f"Length mismatch between cuids ({len(cuids)}) and entity names ({len(names_by_pos)}). Ensure ent_def matches index build artifacts.")
    examples = min(10, len(texts))
    for ex_i in range(examples):
        mention = texts[ex_i]
        gold_cui = golds[ex_i] if ex_i < len(golds) else ''
        gold_name = ''
        if gold_cui:
            rec = ent_def.get(gold_cui)
            if rec:
                gold_name = rec.get('name', '')
        pos_list = topk_unique_pos_by_cui(idx[ex_i], cuids, 10)
        print(f"\nExample {ex_i+1}: {mention}  |  gold: {gold_cui} ({gold_name})")
        for rank, pos in enumerate(pos_list, 1):
            name = names_by_pos[pos]
            cui = cuids[pos]
            print(f"  {rank:>2}. {name} (CUI={cui})")

    # Deduplicate per query to K unique CUIs
    preds_topk: List[List[str]] = []
    for row in idx:
        unique = dedup_topk_by_cui(row, cuids, K)
        preds_topk.append(unique)

    # Accuracy metrics
    acc1 = compute_accuracy(golds, preds_topk, k=1)
    acc5 = compute_accuracy(golds, preds_topk, k=min(5, K))
    acc10 = compute_accuracy(golds, preds_topk, k=min(10, K))
    acc20 = compute_accuracy(golds, preds_topk, k=min(20, K))
    acc40 = compute_accuracy(golds, preds_topk, k=min(40, K))

    result = {
        'num_mentions': len(golds),
        'K_requested': K,
        'K_prime_used': Kprime,
        'faiss_accuracy@1': float(acc1),
        'faiss_accuracy@5': float(acc5),
        'faiss_accuracy@10': float(acc10),
        'faiss_accuracy@20': float(acc20),
        'faiss_accuracy@40': float(acc40),
        'used_cuda_for_encoding': (device.type == 'cuda'),
        'amp': bool(args.amp),
        'auto_batch': bool(args.auto_batch),
        'faiss_gpu': bool(args.use_faiss_gpu),
        'search_batch_size': int(args.search_batch_size or 0),
        'faiss_fp16_index': bool(getattr(args, 'faiss_fp16_index', False)),
        'faiss_temp_mem_mb': int(getattr(args, 'faiss_temp_mem_mb', 0)),
        'unique_within_kprime_mean': (float(diag_mean_unique) if diag_mean_unique is not None else None),
        'unique_within_kprime_median': (int(diag_median_unique) if diag_median_unique is not None else None),
        'pct_queries_with_unique_lt_K': (float(diag_pct_lt_k) if diag_pct_lt_k is not None else None),
    }
    print(json.dumps(result, indent=2))

    # Build and write candidate_data JSON (no extra prints)
    out_path = args.candidates_json_out
    suffix = 'debug' if args.debug else 'full'
    default_out = os.path.join('cross_encoder', f'candidates_{args.split}_J{K}_{suffix}.jsonl.gz')
    if out_path is None or not (out_path.endswith('.jsonl') or out_path.endswith('.jsonl.gz')):
        out_path = default_out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    candidate_data = build_candidates_dict(kept_ids, texts, golds, idx, cuids, names_by_pos, K)
    # Write JSONL (gz-aware), one line per mention, in the original processing order
    is_gz = out_path.endswith('.gz')
    opener = gzip.open if is_gz else open
    num_written = 0
    with opener(out_path, 'wt', encoding='utf-8') as f:
        for mention_id in kept_ids:
            payload = candidate_data[str(mention_id)]
            # Emit mention_id first by constructing an ordered dict-like literal
            ordered = {
                'mention_id': str(mention_id),
                'mention': payload.get('mention', ''),
                'gold_cui': payload.get('gold_cui', ''),
                'candidates': payload.get('candidates', []),
            }
            f.write(json.dumps(ordered, ensure_ascii=False) + "\n")
            num_written += 1
    print(f"Wrote {num_written} lines to {out_path} (JSONL; one JSON object per line).")


def search_in_chunks(index: faiss.Index, queries: np.ndarray, kprime: int, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    if batch_size <= 0:
        raise ValueError(f"search_batch_size must be positive, got {batch_size}")
    scores_list = []
    idx_list = []
    for start in range(0, queries.shape[0], batch_size):
        q = queries[start:start + batch_size]
        s, i = index.search(q, kprime)
        scores_list.append(s)
        idx_list.append(i)
    return np.vstack(scores_list), np.vstack(idx_list)


if __name__ == '__main__':
    main()


# python -m cross_encoder.eval_faiss_candidates_dump --split train --K 10 --batch_size 32 --overshoot_mult 5.0 --debug --use_faiss_gpu --faiss_temp_mem_mb 128 --search_batch_size 1024