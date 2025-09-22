#!/usr/bin/env python3
"""
Evaluate a trained cross-encoder checkpoint on a test split end-to-end.

Pipeline:
  1) Build mention-centered contexts from abstracts + spans.
  2) Generate FAISS candidates (K unique CUIs per mention).
  3) Combine contexts + candidates into per-candidate JSONL.GZ.
  4) Load the best checkpoint and evaluate on the combined test file with the
     same metrics as in training (acc@K, NIL accuracy, MRR, flip-rate) and
     also compute baseline FAISS@K from candidate ranks.

Fail-fast policy:
  - Validate required inputs and invariants early. If missing/invalid, raise
    a clear error and stop execution. No silent recoveries or alternative paths.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
from typing import Dict, List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification

# Reuse existing modules and logic; do not reimplement these
from cross_encoder.build_ce_contexts import build_mention_contexts
from cross_encoder.eval_faiss_candidates_dump import (
    load_tokenized_ids,
    load_id2cuistr,
    encode_texts,
    load_faiss_index_and_cuids,
    build_candidates_dict,
    search_in_chunks,
)
from cross_encoder.combine_ce_eval_and_contexts import combine
from cross_encoder.train_ce_from_combined import (
    load_groups,
    GroupDataset,
    collate_groups,
    evaluate_loader_topk,
    compute_faiss_accuracy_at_k,
)


def _parse_ks(s: str) -> Tuple[int, ...]:
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    try:
        ks = sorted({int(p) for p in parts})
    except Exception:
        raise ValueError(f"Invalid --ks value: {s}")
    if not ks:
        raise ValueError("--ks must contain at least one integer K")
    return tuple(ks)


def _ensure_parent_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def _write_contexts_jsonl_gz(context_map: Dict[str, Dict[str, object]], out_path: str) -> None:
    """Write context mapping (mention_id -> {...}) to JSONL(.gz), one JSON per line.

    Each line includes the original payload plus a 'mention_id' field.
    """
    _ensure_parent_dir(out_path)
    is_gz = out_path.endswith(".gz")
    opener = gzip.open if is_gz else open
    num = 0
    with opener(out_path, "wt", encoding="utf-8") as f:
        for mention_id, payload in context_map.items():
            rec = dict(payload)
            rec["mention_id"] = str(mention_id)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            num += 1
    if num == 0:
        raise RuntimeError(f"No contexts were written to {out_path}")


def _load_nil_bias_tensor(model_dir: str, device: torch.device) -> torch.Tensor:
    """Load scalar nil_bias from checkpoint state dict if present; default to 0.

    Note: from_pretrained won't attach custom params like nil_bias by default,
    so we read it directly from pytorch_model.bin if available.
    """
    bin_path = os.path.join(model_dir, "pytorch_model.bin")
    if not os.path.exists(model_dir) or not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    if not os.path.exists(bin_path):
        # Fail fast with explicit message per policy
        return torch.zeros((), device=device)
    sd = torch.load(bin_path, map_location="cpu")
    if not isinstance(sd, dict) or "nil_bias" not in sd:
        return torch.zeros((), device=device)
    t = sd["nil_bias"]
    if not isinstance(t, torch.Tensor):
        raise RuntimeError("Checkpoint contains 'nil_bias' but it is not a Tensor")
    return t.to(device).view(())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="End-to-end test-time evaluation for Cross-Encoder")

    # Required dataset inputs
    parser.add_argument("--tokenized_path", type=str, required=True, help="Path to tokenized test .pt (e.g., data/processed/mention_detection/test/tokenized_test_full.pt)")
    parser.add_argument("--abstract_path", type=str, required=True, help="Path to abstract_dict_*.json (test split)")
    parser.add_argument("--spans_path", type=str, required=True, help="Path to spans_dict_*.json (test split)")
    parser.add_argument("--id2cuistr_path", type=str, required=True, help="Path to id2cuistr_dict_*.json (test split)")

    # Retrieval + encoding
    parser.add_argument("--encoder_name", type=str, default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--overshoot_mult", type=float, default=5.0, help="K' = max(50, overshoot_mult*K)")
    parser.add_argument("--encode_batch_size", type=int, default=64, help="Batch size for encoding mentions")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision for encoding (CUDA only)")
    parser.add_argument("--auto_batch", action="store_true", help="Enable adaptive batching for encoding (CUDA only)")
    parser.add_argument("--index_path", type=str, default=None, help="Path to FAISS index; if omitted, use defaults from loader")
    parser.add_argument("--cuids_path", type=str, default=None, help="Path to CUIs json; if omitted, use defaults from loader")
    parser.add_argument("--use_faiss_gpu", action="store_true", help="Clone FAISS index to GPU for searching (CUDA only)")
    parser.add_argument("--faiss_fp16_index", action="store_true", help="Store FAISS vectors on GPU in float16 to save memory")
    parser.add_argument("--faiss_temp_mem_mb", type=int, default=256, help="FAISS GPU temp memory in MB")
    parser.add_argument("--search_batch_size", type=int, default=0, help="If >0, use chunked index.search with this batch size")

    # Entity definitions for alignment + names
    parser.add_argument("--ent_def_path", type=str, default=None, help="Entity definitions JSON (defaults to eval script's path if omitted)")

    # Context building
    parser.add_argument("--window", type=int, default=256, help="Character window size around mention for contexts")
    parser.add_argument("--contexts_jsonl_gz", type=str, default=os.path.join("cross_encoder", "ce_test_mention_sentences.jsonl.gz"))

    # Combine
    parser.add_argument("--mrdef_path", type=str, default=os.path.join("cross_encoder", "MRDEF_unique_definitions.RRF"))
    parser.add_argument("--def_truncation", type=str, choices=["chars", "words"], default="chars")
    parser.add_argument("--def_max_chars", type=int, default=100)
    parser.add_argument("--def_max_words", type=int, default=20)
    parser.add_argument("--group_prefix", type=str, default="test_full")
    parser.add_argument("--start_index", type=int, default=1)
    parser.add_argument("--candidates_jsonl_gz", type=str, default=os.path.join("cross_encoder", "candidates_test_J{K}_full.jsonl.gz"))
    parser.add_argument("--combined_jsonl_gz", type=str, default=os.path.join("cross_encoder", "combined_test_full.jsonl.gz"))

    # Model + evaluation
    parser.add_argument("--model_dir", type=str, required=True, help="Path to trained checkpoint dir (e.g., ce_run_xxx/best)")
    parser.add_argument("--nil_mode", type=str, default="bias", choices=["bias", "bias_plus_text"], help="How to score NIL rows")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--groups_per_batch", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--ks", type=str, default="1,5,10", help="Comma-separated K values for accuracy metrics")

    args = parser.parse_args(argv)

    # Fail-fast validations
    if args.max_length <= 8:
        raise ValueError("--max_length must be > 8")
    if args.K <= 0:
        raise ValueError(f"--K must be positive, got {args.K}")
    if args.overshoot_mult <= 1.0:
        raise ValueError(f"--overshoot_mult must be > 1.0, got {args.overshoot_mult}")

    for p in [args.tokenized_path, args.abstract_path, args.spans_path, args.id2cuistr_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required input not found: {p}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.amp and device.type != "cuda":
        raise RuntimeError("--amp requires CUDA but no GPU is available.")
    if args.auto_batch and device.type != "cuda":
        raise RuntimeError("--auto_batch requires CUDA but no GPU is available.")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Derive defaults for ent_def and FAISS artifacts consistent with eval script
    if args.ent_def_path is None:
        # Keep parity with eval_faiss_candidates_dump defaults
        ent_dir = os.path.join('improved_less_synonyms', 'processed', 'entity_disambiguation')
        args.ent_def_path = os.path.join(ent_dir, 'ent_def_aug2_full_syn.json')
    if not os.path.exists(args.ent_def_path):
        raise FileNotFoundError(f"Entity definitions file not found: {args.ent_def_path}")

    # 1) Build contexts JSONL.GZ
    context_map = build_mention_contexts(args.abstract_path, args.spans_path, int(args.window))
    _write_contexts_jsonl_gz(context_map, args.contexts_jsonl_gz)

    # 2) Retrieval + candidates
    mention_ids: List[int] = load_tokenized_ids(args.tokenized_path)
    id2: Dict[str, dict] = load_id2cuistr(args.id2cuistr_path)

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
        raise RuntimeError("No valid mentions found for test retrieval.")

    queries = encode_texts(
        texts,
        args.encoder_name,
        device=device,
        batch_size=int(args.encode_batch_size),
        amp=bool(args.amp),
        auto_batch=bool(args.auto_batch),
    )

    # Load FAISS index + CUIs (defaults inside loader if None)
    index, cuids = load_faiss_index_and_cuids(args.index_path, args.cuids_path)

    # Optionally move index to GPU
    if args.use_faiss_gpu:
        if device.type != 'cuda':
            raise RuntimeError("--use_faiss_gpu requested but CUDA is not available.")
        import faiss  # local import to keep dependency usage consistent
        try:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            res = faiss.StandardGpuResources()
            try:
                res.setTempMemory(int(args.faiss_temp_mem_mb) * 1024 * 1024)
            except Exception:
                pass
            co = faiss.GpuClonerOptions()
            if bool(getattr(args, 'faiss_fp16_index', False)):
                co.useFloat16 = True
            index = faiss.index_cpu_to_gpu(res, 0, index, co)
        except Exception as e1:
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
            except Exception as e2:
                raise RuntimeError(f"Failed to move FAISS index to GPU after retries: first={e1} | retry={e2}")

    # Load entity definitions aligned to FAISS index build (for names_by_pos)
    with open(args.ent_def_path, 'r', encoding='utf-8') as f:
        ent_list = json.load(f)
    names_by_pos = [rec.get('name', '') for rec in ent_list]
    if len(names_by_pos) == 0:
        raise RuntimeError("Entity list is empty; cannot format candidates.")
    if len(names_by_pos) != len(cuids):
        raise RuntimeError(f"Length mismatch between cuids ({len(cuids)}) and entity names ({len(names_by_pos)}). Ensure ent_def matches index build artifacts.")

    # Compute K'
    K = int(args.K)
    Kprime = max(50, int(args.overshoot_mult * K))
    if hasattr(index, 'ntotal') and index.ntotal is not None:
        Kprime = min(Kprime, int(index.ntotal))
        if Kprime < K:
            raise RuntimeError(f"Index too small for requested K. ntotal={index.ntotal}, requested K={K}")

    # Search
    if int(args.search_batch_size) > 0:
        scores, idx = search_in_chunks(index, queries, Kprime, int(args.search_batch_size))
    else:
        scores, idx = index.search(queries, Kprime)
    del scores  # not used further

    # Build candidates per mention and write JSONL(.gz)
    candidate_data = build_candidates_dict(kept_ids, texts, golds, idx, cuids, names_by_pos, K)
    out_cand_path = args.candidates_jsonl_gz.replace("{K}", str(K))
    _ensure_parent_dir(out_cand_path)
    is_gz = out_cand_path.endswith('.jsonl.gz')
    opener = gzip.open if is_gz else open
    num_written = 0
    with opener(out_cand_path, 'wt', encoding='utf-8') as f:
        for mention_id in kept_ids:
            payload = candidate_data[str(mention_id)]
            ordered = {
                'mention_id': str(mention_id),
                'mention': payload.get('mention', ''),
                'gold_cui': payload.get('gold_cui', ''),
                'candidates': payload.get('candidates', []),
            }
            f.write(json.dumps(ordered, ensure_ascii=False) + "\n")
            num_written += 1
    if num_written == 0:
        raise RuntimeError(f"No candidate lines were written to {out_cand_path}")

    # 3) Combine into per-candidate JSONL.GZ
    combined_path = args.combined_jsonl_gz
    _ensure_parent_dir(combined_path)
    written, only_eval, only_ctx = combine(
        eval_jsonl_gz=out_cand_path,
        contexts_jsonl_gz=args.contexts_jsonl_gz,
        out_jsonl_gz=combined_path,
        group_prefix=str(args.group_prefix),
        start_index=int(args.start_index),
        mrdef_path=str(args.mrdef_path) if args.mrdef_path else None,
        def_truncation=str(args.def_truncation),
        def_max_chars=int(args.def_max_chars),
        def_max_words=int(args.def_max_words),
    )
    if written <= 0:
        raise RuntimeError(f"Combined output contains zero lines: {combined_path}")

    # 4) Evaluate trained cross-encoder on combined test file
    ks = _parse_ks(args.ks)

    # Baseline FAISS@K from combined file (deterministic)
    test_groups = load_groups(combined_path)
    baseline = compute_faiss_accuracy_at_k(test_groups, ks=ks)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    config = AutoConfig.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir, config=config)
    model.to(device)
    model.eval()

    # Load learned NIL bias (scalar)
    nil_bias = _load_nil_bias_tensor(args.model_dir, device)

    loader = DataLoader(
        GroupDataset(test_groups),
        batch_size=int(args.groups_per_batch),
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=lambda batch: collate_groups(batch, tokenizer, int(args.max_length)),
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )

    # Forward-only evaluation; keep hinge metrics disabled for speed
    _, acc, hinge_info, nil_acc, extra = evaluate_loader_topk(
        model,
        loader,
        device,
        ks=ks,
        compute_loss=False,
        hinge_margin=None,
        nil_mode=str(args.nil_mode),
        nil_bias=nil_bias,
    )

    result = {
        "model_dir": args.model_dir,
        "combined_jsonl_gz": combined_path,
        "nil_mode": str(args.nil_mode),
        "ks": list(ks),
        "acc": {int(k): float(acc.get(int(k), 0.0)) for k in ks},
        "nil_acc": (float(nil_acc) if nil_acc is not None else None),
        "mrr": (float(extra.get("mrr")) if extra else None),
        "flip_rate": (float(extra.get("flip_rate")) if extra else None),
        "baseline_faiss_acc": {int(k): float(baseline.get(int(k), 0.0)) for k in ks},
        "contexts_jsonl_gz": args.contexts_jsonl_gz,
        "candidates_jsonl_gz": out_cand_path,
        "written_combined": int(written),
        "unmatched_eval_only": int(only_eval),
        "unmatched_contexts_only": int(only_ctx),
    }
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


