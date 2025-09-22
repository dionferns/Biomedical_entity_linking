import os
import json
import gzip
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import DataLoader
from transformers import (
    BertModel, BertTokenizerFast,
    AutoTokenizer, AutoConfig, AutoModelForSequenceClassification,
)
import sys

# Ensure project root on sys.path for intra-project imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_processing.mention_detection.md_preprocessing_v2 import MentionDetectionPreprocessingV2
from models.mention_detection import MentionDetection
from utils.metrics_tracker import MetricsTracker
from utils.error_handling import safe_json_load
from utils.md_span_metrics import decode_bio_to_token_spans, prf

# CE utilities (reused from CE evaluation). 
from cross_encoder.build_ce_contexts import build_mention_contexts
from cross_encoder.eval_faiss_candidates_dump import (
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


def _fail_if_missing(paths: List[str], context: str) -> None:
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required {context}: {p}")


def _write_jsonl(path: str, rows: List[dict]) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        n = 0
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    if n == 0:
        raise RuntimeError(f"No lines written to {path}")


def _write_jsonl_gz(path: str, rows: List[dict]) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "wt", encoding="utf-8") as f:
        n = 0
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    if n == 0:
        raise RuntimeError(f"No lines written to {path}")


def _write_contexts_jsonl_gz(context_map: Dict[str, Dict[str, object]], out_path: str) -> None:
    d = os.path.dirname(out_path)
    if d:
        os.makedirs(d, exist_ok=True)
    opener = gzip.open if out_path.endswith(".gz") else open
    with opener(out_path, "wt", encoding="utf-8") as f:
        n = 0
        for mention_id, payload in context_map.items():
            rec = dict(payload)
            rec["mention_id"] = str(mention_id)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    if n == 0:
        raise RuntimeError(f"No contexts were written to {out_path}")


def _resolve_md_ckpt(debug: bool, ckpt_path: Optional[str], save_models_dir_name: Optional[str]) -> str:
    if ckpt_path:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Provided MD checkpoint does not exist: {ckpt_path}")
        return ckpt_path
    subdir = (save_models_dir_name.strip() if save_models_dir_name else "md")
    debug_suffix = "debug" if debug else "full"
    candidate = os.path.join("saved_models", subdir, f"best_{debug_suffix}.pt")
    if not os.path.exists(candidate):
        raise FileNotFoundError(
            f"Default MD checkpoint not found: {candidate}. Provide --md_ckpt_path or ensure best symlink exists."
        )
    return candidate


def main(
    md_ckpt_path: Optional[str] = None,
    save_models_dir_name: Optional[str] = None,
    debug: bool = True,
    batch_size: int = 8,
    num_workers: int = 4,
    metrics_dir_name: Optional[str] = None,
    # MD span matching
    lenient_mode: str = 'any_overlap',
    iou_threshold: float = 0.5,
    tail_window: int = 64,
    # Processed test JSONs
    abstract_path: Optional[str] = None,
    spans_path: Optional[str] = None,
    # FAISS artifacts
    faiss_index_path: str = "improved_less_synonyms/processed/entity_disambiguation/faiss_index_aug2_name_full.bin",
    cuids_path: str = "improved_less_synonyms/processed/entity_disambiguation/cuids_aug2_name_full.json",
    # CE model + definitions
    ce_model_dir: str = "",
    ent_def_path: str = "improved_less_synonyms/processed/entity_disambiguation/ent_def_aug2_full_syn.json",
    # Retrieval/contexts
    K: int = 10,
    overshoot_mult: float = 5.0,
    search_batch_size: int = 0,
    contexts_window: int = 256,
    # Outputs (all artifacts will be written ONLY under this folder)
    out_dir: str = "results/full_el_test",
) -> Dict[str, object]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    debug_suffix = "debug" if debug else "full"
    # Fail-fast on pre-existing output folder to avoid accidental overwrite
    if os.path.exists(out_dir):
        raise FileExistsError(
            f"Output folder already exists: {out_dir}. Please choose a different folder name via --out_dir."
        )
    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO] Output directory: {out_dir}")

    # Validate CE and FAISS inputs
    if not ce_model_dir:
        raise ValueError("--ce_model_dir is required")
    _fail_if_missing([ce_model_dir], "CE model dir")
    _fail_if_missing([faiss_index_path, cuids_path, ent_def_path], "FAISS/defs input")

    # Prepare MD preprocessing object and processed paths
    md = MentionDetectionPreprocessingV2(debug=debug, split_type="test")
    abs_json = abstract_path or md.abstract_dict_filepath
    sp_json = spans_path or md.spans_dict_filepath
    _fail_if_missing([abs_json, sp_json], "processed test JSON")
    print(f"[INFO] Using abstracts: {abs_json}")
    print(f"[INFO] Using gold spans: {sp_json}")
    print(f"[INFO] CE model dir: {ce_model_dir}")
    print(f"[INFO] FAISS index: {faiss_index_path}")
    print(f"[INFO] CUIs file: {cuids_path}")

    # Require pre-tokenized test tensors; do NOT build them here (fail-fast policy)
    tokenizer = BertTokenizerFast.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    if not os.path.exists(md.tokenized_filepath):
        raise FileNotFoundError(
            f"Tokenized test tensors not found: {md.tokenized_filepath}. "
            f"Please run preprocessing to generate them before running this test script."
        )
    test_data = torch.load(md.tokenized_filepath)
    print(f"[INFO] Using tokenized test tensors: {md.tokenized_filepath}")

    # DataLoader
    test_loader = MentionDetectionPreprocessingV2.create_dataloader(
        test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    abstracts = safe_json_load(abs_json)
    gold_spans = safe_json_load(sp_json)

    # Load MD checkpoint and run inference
    enc = BertModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext").to(device).eval()
    head = MentionDetection(num_labels=3, dropout=0.1, hidden_size=enc.config.hidden_size).to(device).eval()
    md_ckpt = _resolve_md_ckpt(debug, md_ckpt_path, save_models_dir_name)
    sd = torch.load(md_ckpt, map_location=device)
    if 'encoder_state_dict' not in sd or 'mention_head_state_dict' not in sd:
        raise KeyError(f"MD checkpoint missing required keys: {md_ckpt}")
    enc.load_state_dict(sd['encoder_state_dict'])
    head.load_state_dict(sd['mention_head_state_dict'])

    # Build pmid index mapping for each row in loader order
    pmids_by_row: List[str] = []
    for pmid, rows in test_data.items():
        for _ in rows:
            pmids_by_row.append(pmid)

    all_logits, all_labels, all_attn = [], [], []
    with torch.no_grad():
        for input_ids, attention_mask, labels, _ in test_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            seq_out = enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            _, logits = head(seq_out, ner_labels=labels)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            all_attn.append(attention_mask.cpu())

    if not all_logits:
        raise RuntimeError("No test batches processed. Check tokenized test tensors and loader config.")

    # MD metrics
    cat_logits = torch.cat(all_logits, dim=0)
    cat_labels = torch.cat(all_labels, dim=0)
    cat_attn = torch.cat(all_attn, dim=0)
    tracker = MetricsTracker(experiment_name=metrics_dir_name or f"md_full_{debug_suffix}")
    md_metrics = tracker.calculate_md_token_f1(cat_logits, cat_labels)
    num_tokens = int((cat_labels != -100).sum().item())
    # MD token summary print (akin to test_md_only)
    md_p = float(md_metrics.get("md_token_precision", md_metrics.get("precision", 0.0)))
    md_r = float(md_metrics.get("md_token_recall", md_metrics.get("recall", 0.0)))
    md_f1 = float(md_metrics.get("md_token_f1", md_metrics.get("f1", 0.0)))
    print(f"[MD] Token F1: {md_f1:.3f} | P/R: {md_p:.3f}/{md_r:.3f} | tokens: {num_tokens}")

    # Extract predicted spans + string mentions + gold_cui via alignment
    preds_all = cat_logits.argmax(dim=-1)
    total_rows = cat_labels.size(0)
    if len(pmids_by_row) != total_rows:
        raise RuntimeError(f"pmid index mapping size mismatch (pmids={len(pmids_by_row)} vs rows={total_rows})")

    mention_rows: List[Dict[str, object]] = []
    per_pmid_mentions: Dict[str, List[Dict[str, object]]] = {}
    mention_id = 1

    for i in range(total_rows):
        pmid = pmids_by_row[i]
        labs_i = cat_labels[i].tolist()
        preds_i = preds_all[i].tolist()
        attn_i = cat_attn[i]
        t_eff = int(attn_i.sum().item())
        valid = [(x != -100) for x in labs_i]

        pred_tok = decode_bio_to_token_spans(preds_i, valid, ignore_index=-100)

        if pmid not in abstracts:
            raise KeyError(f"Missing pmid in abstracts: {pmid}")
        text = abstracts[pmid]
        enc_abs = tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=t_eff)
        offsets = enc_abs.get('offset_mapping', [])
        if not offsets or len(offsets) != t_eff:
            raise RuntimeError(
                f"Offset mapping length mismatch for pmid={pmid}: expected {t_eff}, got {len(offsets) if offsets else 0}"
            )

        # Build gold char spans → cui map from processed gold spans JSON
        gold_char_to_cui = {(int(s['start']), int(s['end'])): s.get('cui', 'NIL') for s in gold_spans.get(pmid, [])}

        for (s_tok, e_tok) in pred_tok:
            s_char = int(offsets[s_tok][0])
            e_char = int(offsets[e_tok - 1][1])
            if e_char <= s_char:
                continue
            mtext = text[s_char:e_char]

            # Assign gold_cui by overlap/IoU
            gold_cui = "NIL"
            for (gs, ge), cui in gold_char_to_cui.items():
                inter = max(0, min(e_char, ge) - max(s_char, gs))
                if lenient_mode == 'any_overlap':
                    if inter > 0:
                        gold_cui = cui
                        break
                else:
                    union = (e_char - s_char) + (ge - gs) - inter
                    if union > 0 and (inter / union) >= float(iou_threshold):
                        gold_cui = cui
                        break

            rec = {
                "mention_id": str(mention_id),
                "pmid": pmid,
                "start": int(s_char),
                "end": int(e_char),
                "text": mtext,
                "gold_cui": gold_cui,
            }
            mention_rows.append(rec)
            per_pmid_mentions.setdefault(pmid, []).append({"start": rec["start"], "end": rec["end"], "text": mtext})
            mention_id += 1

    if not mention_rows:
        raise RuntimeError("No predicted mentions extracted; cannot proceed to retrieval.")
    # Counts and sample
    unique_pmids = len(set(pmids_by_row))
    print(f"[MD] Rows={total_rows}, predicted_mentions={len(mention_rows)}, unique_PMIDs={unique_pmids}")
    for i, rec in enumerate(mention_rows[:3]):
        preview = rec["text"][:60].replace("\n", " ")
        print(f"[MD] Sample mention {i+1}: pmid={rec['pmid']} span=[{rec['start']},{rec['end']}] text='{preview}'")

    # Save MD metrics + predicted mentions per abstract
    # Write MD metrics inside the unified output folder
    test_metrics_path = os.path.join(out_dir, "md_test_metrics.jsonl")
    md_record = {
        "split": "test",
        "md_token_f1": float(md_metrics.get("md_token_f1", md_metrics.get("f1", 0.0))),
        "md_token_precision": float(md_metrics.get("md_token_precision", md_metrics.get("precision", 0.0))),
        "md_token_recall": float(md_metrics.get("md_token_recall", md_metrics.get("recall", 0.0))),
        "num_tokens": int(num_tokens),
        "lenient_mode": str(lenient_mode),
        "iou_threshold": float(iou_threshold),
    }
    with open(test_metrics_path, "a", encoding="utf-8") as f:
        json.dump(md_record, f); f.write("\n")

    predicted_mentions_out = os.path.join(out_dir, f"predicted_mentions_{debug_suffix}.jsonl.gz")
    # Limit to the first 100 PMIDs for this preview file
    pmids_list = list(abstracts.keys())[:100]
    rows = [{"pmid": pmid, "predicted_mentions": per_pmid_mentions.get(pmid, [])} for pmid in pmids_list]
    _write_jsonl_gz(predicted_mentions_out, rows)

    # 3) Retrieval on predicted mentions (SapBERT → FAISS)
    texts = [r["text"] for r in mention_rows]
    golds = [r["gold_cui"] for r in mention_rows]
    kept_ids = [r["mention_id"] for r in mention_rows]
    if not texts:
        raise RuntimeError("No predicted mention texts for retrieval.")
    queries = encode_texts(
        texts,
        "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        device=device,
        batch_size=64,
        amp=False,
        auto_batch=False,
    )
    index, cuids = load_faiss_index_and_cuids(faiss_index_path, cuids_path)

    Kprime = max(50, int(overshoot_mult * K))
    if hasattr(index, 'ntotal') and index.ntotal is not None:
        Kprime = min(Kprime, int(index.ntotal))
        if Kprime < K:
            raise RuntimeError(f"Index too small for requested K. ntotal={index.ntotal}, requested K={K}")
    if int(search_batch_size) > 0:
        scores, idx = search_in_chunks(index, queries, Kprime, int(search_batch_size))
    else:
        scores, idx = index.search(queries, Kprime)
    del scores
    print(f"[RET] Retrieval: mentions={len(kept_ids)}, K={K}, K'={Kprime}, index.ntotal={getattr(index,'ntotal','NA')}")

    with open(ent_def_path, "r", encoding="utf-8") as f:
        ent_list = json.load(f)
    names_by_pos = [rec.get("name", "") for rec in ent_list]
    if len(names_by_pos) == 0:
        raise RuntimeError("Entity definitions list is empty.")
    if len(names_by_pos) != len(cuids):
        raise RuntimeError(f"Length mismatch: cuids={len(cuids)} vs entity names={len(names_by_pos)}")

    cand_map = build_candidates_dict(kept_ids, texts, golds, idx, cuids, names_by_pos, K)
    candidates_jsonl_gz = os.path.join(out_dir, f"candidates_test_pred_J{K}_{debug_suffix}.jsonl.gz")
    _write_jsonl_gz(candidates_jsonl_gz, [
        {
            "mention_id": mid,
            "mention": cand_map[str(mid)]["mention"] if isinstance(mid, str) else cand_map[mid]["mention"],
            "gold_cui": cand_map[str(mid)]["gold_cui"] if isinstance(mid, str) else cand_map[mid]["gold_cui"],
            "candidates": cand_map[str(mid)]["candidates"] if isinstance(mid, str) else cand_map[mid]["candidates"],
        }
        for mid in kept_ids
    ])

    # 4) Build contexts from predicted spans
    spans_pred_json = os.path.join(out_dir, f"predicted_spans_{debug_suffix}.json")
    tmp_spans_map: Dict[str, List[Dict[str, object]]] = {}
    for r in mention_rows:
        tmp_spans_map.setdefault(r["pmid"], []).append({
            "id": int(r["mention_id"]),
            "start": int(r["start"]),
            "end": int(r["end"]),
            "mention": r["text"],
            "cui": r["gold_cui"],  # ADD THIS LINE

        })
    # _write_jsonl(spans_pred_json, [])  # ensure parent dir exists via writer

    os.makedirs(os.path.dirname(spans_pred_json), exist_ok=True)
    with open(spans_pred_json, "w", encoding="utf-8") as f:
        json.dump(tmp_spans_map, f, ensure_ascii=False, indent=2)

    contexts_map = build_mention_contexts(abs_json, spans_pred_json, int(contexts_window))
    contexts_jsonl_gz = os.path.join(out_dir, f"ce_test_pred_mention_sentences_{debug_suffix}.jsonl.gz")
    _write_contexts_jsonl_gz(contexts_map, contexts_jsonl_gz)
    print(f"[CTX] Contexts built: {len(contexts_map)} -> {contexts_jsonl_gz}")

    # 5) Combine + CE reranking + evaluation
    combined_jsonl_gz = os.path.join(out_dir, f"combined_test_pred_{debug_suffix}.jsonl.gz")
    written, only_eval, only_ctx = combine(
        eval_jsonl_gz=candidates_jsonl_gz,
        contexts_jsonl_gz=contexts_jsonl_gz,
        out_jsonl_gz=combined_jsonl_gz,
        group_prefix=f"test_pred_{debug_suffix}",
        start_index=1,
        mrdef_path=os.path.join("cross_encoder", "MRDEF_unique_definitions.RRF"),
        def_truncation="words",
        def_max_chars=100,
        def_max_words=20,
    )
    if written <= 0:
        raise RuntimeError(f"Combined output contains zero lines: {combined_jsonl_gz}")
    print(f"[COMBINE] groups={written}, unmatched eval-only={only_eval}, unmatched ctx-only={only_ctx}")

    # Baseline FAISS acc@K
    test_groups = load_groups(combined_jsonl_gz)
    baseline = compute_faiss_accuracy_at_k(test_groups, ks=(1, 5, 10, 20))
    print(
        f"[FAISS] baseline acc@1/5/10/20: "
        f"{float(baseline.get(1,0.0)):.3f}/{float(baseline.get(5,0.0)):.3f}/"
        f"{float(baseline.get(10,0.0)):.3f}/{float(baseline.get(20,0.0)):.3f}"
    )

    # CE model
    tokenizer_ce = AutoTokenizer.from_pretrained(ce_model_dir)
    config_ce = AutoConfig.from_pretrained(ce_model_dir)
    model_ce = AutoModelForSequenceClassification.from_pretrained(ce_model_dir, config=config_ce)
    model_ce.to(device); model_ce.eval()
    # Optional NIL bias if present
    bin_path = os.path.join(ce_model_dir, "pytorch_model.bin")
    nil_bias = torch.zeros((), device=device)
    if os.path.exists(bin_path):
        try:
            sd = torch.load(bin_path, map_location="cpu")
            if isinstance(sd, dict) and "nil_bias" in sd and isinstance(sd["nil_bias"], torch.Tensor):
                nil_bias = sd["nil_bias"].to(device).view(())
        except Exception:
            # Fail-fast: do not mask CE eval, just proceed with zero bias if loading extra tensor fails
            print("Missing or invalid nil_bias in CE checkpoint")  # BE AWARE OF THIS
            nil_bias = torch.zeros((), device=device)

    loader = DataLoader(
        GroupDataset(test_groups),
        batch_size=32,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_groups(batch, tokenizer_ce, 128),
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )

    _, acc, _, nil_acc, extra = evaluate_loader_topk(
        model_ce,
        loader,
        device,
        ks=(1, 5, 10),
        compute_loss=False,
        hinge_margin=None,
        nil_mode="bias",
        nil_bias=nil_bias,
    )
    ce_acc1 = float(acc.get(1, 0.0)); ce_acc5 = float(acc.get(5, 0.0)); ce_acc10 = float(acc.get(10, 0.0))
    mrr = float(extra.get("mrr", 0.0)) if extra else 0.0
    nil_print = (float(nil_acc) if nil_acc is not None else 0.0)
    print(f"[CE] acc@1/5/10: {ce_acc1:.3f}/{ce_acc5:.3f}/{ce_acc10:.3f} | MRR: {mrr:.3f} | NIL acc: {nil_print:.3f}")

    result = {
        "md_metrics": md_record,
        "predicted_mentions_path": predicted_mentions_out,
        "candidates_jsonl_gz": candidates_jsonl_gz,
        "contexts_jsonl_gz": contexts_jsonl_gz,
        "combined_jsonl_gz": combined_jsonl_gz,
        "baseline_faiss_acc": {int(k): float(v) for k, v in baseline.items()},
        "ce_acc": {int(k): float(acc.get(int(k), 0.0)) for k in (1, 5, 10)},
        "nil_acc": (float(nil_acc) if nil_acc is not None else None),
        "mrr": (float(extra.get("mrr")) if extra else None),
        "flip_rate": (float(extra.get("flip_rate")) if extra else None),
        "unmatched_eval_only": int(only_eval),
        "unmatched_contexts_only": int(only_ctx),
    }
    # Persist full end-to-end evaluation summary inside out_dir
    final_result_path = os.path.join(out_dir, "final_result.json")
    with open(final_result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result))
    print(f"[DONE] Wrote final summary: {final_result_path}")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Full end-to-end entity linking test (MD → FAISS → CE).")
    parser.add_argument('--md_ckpt_path', type=str, default=None)
    parser.add_argument('--save_models_dir_name', type=str, default=None)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--metrics_dir_name', type=str, default=None)
    parser.add_argument('--lenient_mode', type=str, default='any_overlap', choices=['any_overlap', 'iou'])
    parser.add_argument('--iou_threshold', type=float, default=0.5)
    parser.add_argument('--tail_window', type=int, default=64)
    parser.add_argument('--abstract_path', type=str, default=None)
    parser.add_argument('--spans_path', type=str, default=None)
    parser.add_argument('--faiss_index_path', type=str, default="improved_less_synonyms/processed/entity_disambiguation/faiss_index_aug2_name_full.bin")
    parser.add_argument('--cuids_path', type=str, default="improved_less_synonyms/processed/entity_disambiguation/cuids_aug2_name_full.json")
    parser.add_argument('--ce_model_dir', type=str, required=True)
    parser.add_argument('--ent_def_path', type=str, default="improved_less_synonyms/processed/entity_disambiguation/ent_def_aug2_full_syn.json")
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--overshoot_mult', type=float, default=5.0)
    parser.add_argument('--search_batch_size', type=int, default=0)
    parser.add_argument('--contexts_window', type=int, default=120)
    parser.add_argument('--out_dir', type=str, default="results/full_el_test")
    args = parser.parse_args()
    main(
        md_ckpt_path=args.md_ckpt_path,
        save_models_dir_name=args.save_models_dir_name,
        debug=bool(args.debug),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        metrics_dir_name=args.metrics_dir_name,
        lenient_mode=str(args.lenient_mode),
        iou_threshold=float(args.iou_threshold),
        tail_window=int(args.tail_window),
        abstract_path=args.abstract_path,
        spans_path=args.spans_path,
        faiss_index_path=args.faiss_index_path,
        cuids_path=args.cuids_path,
        ce_model_dir=args.ce_model_dir,
        ent_def_path=args.ent_def_path,
        K=int(args.K),
        overshoot_mult=float(args.overshoot_mult),
        search_batch_size=int(args.search_batch_size),
        contexts_window=int(args.contexts_window),
        out_dir=args.out_dir,
    )


