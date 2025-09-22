import os
import json
from typing import Optional, Dict, List, Tuple
from pathlib import Path

import torch
from transformers import BertModel, BertTokenizerFast
import sys

# Ensure project root on sys.path for intra-project imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_processing.mention_detection.md_preprocessing_v2 import MentionDetectionPreprocessingV2
from models.mention_detection import MentionDetection
from utils.metrics_tracker import MetricsTracker
from utils.error_handling import safe_json_load
from utils.md_span_metrics import decode_bio_to_token_spans, match_spans, prf


def _ensure_test_ready(debug: bool, tokenizer: BertTokenizerFast):
    md = MentionDetectionPreprocessingV2(debug=debug, split_type="test")
    required_path = md.tokenized_filepath
    if not os.path.exists(required_path):
        mode = "debug" if debug else "full"
        raise FileNotFoundError(
            (
                f"Missing required tokenized tensors for mention detection (test split): {required_path}\n"
                f"Mode: '{mode}'.\n"
                f"Please run preprocessing, e.g.:\n"
                f"  python -m data_processing.mention_detection.run_md_preprocessing_v2 --mode {mode} --split test"
            )
        )
    data = torch.load(required_path)
    return data, md


def _resolve_ckpt_path(debug: bool, ckpt_path: Optional[str], save_models_dir_name: Optional[str]) -> str:
    if ckpt_path:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Provided checkpoint path does not exist: {ckpt_path}")
        return ckpt_path
    subdir = (save_models_dir_name.strip() if save_models_dir_name else "md")
    debug_suffix = "debug" if debug else "full"
    candidate = os.path.join("saved_models", subdir, f"best_{debug_suffix}.pt")
    if not os.path.exists(candidate):
        raise FileNotFoundError(
            (
                f"Default checkpoint not found: {candidate}\n"
                f"Provide --ckpt_path explicitly or train and ensure best symlink exists."
            )
        )
    return candidate


def main(
    ckpt_path: Optional[str] = None,
    save_models_dir_name: Optional[str] = None,
    debug: bool = True,
    batch_size: int = 8,
    num_workers: Optional[int] = 4,
    experiment_name: Optional[str] = None,
    metrics_dir_name: Optional[str] = None,
    enable_span_eval: bool = True,
    lenient_mode: str = 'any_overlap',
    iou_threshold: float = 0.5,
    tail_window: int = 64,
) -> Dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    debug_suffix = "debug" if debug else "full"

    # Encoder and tokenizer (SapBERT)
    encoder = BertModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext").to(device)
    tokenizer = BertTokenizerFast.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")

    mention_head = MentionDetection(
        num_labels=3,
        dropout=0.1,
        hidden_size=encoder.config.hidden_size,
    ).to(device)

    # Checkpoint
    ckpt_resolved = _resolve_ckpt_path(debug, ckpt_path, save_models_dir_name)
    ckpt = torch.load(ckpt_resolved, map_location=device)
    if 'encoder_state_dict' not in ckpt or 'mention_head_state_dict' not in ckpt:
        raise KeyError(f"Checkpoint missing required keys: {ckpt_resolved}")
    encoder.load_state_dict(ckpt['encoder_state_dict'])
    mention_head.load_state_dict(ckpt['mention_head_state_dict'])
    encoder.eval(); mention_head.eval()

    # Data
    test_data, md_test = _ensure_test_ready(debug, tokenizer)
    test_loader = MentionDetectionPreprocessingV2.create_dataloader(
        test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # Metrics dir
    metrics_subdir = metrics_dir_name.strip() if metrics_dir_name else f"md_test_{debug_suffix}"
    metrics_dir = os.path.join("results", metrics_subdir)
    os.makedirs(metrics_dir, exist_ok=True)
    test_metrics_path = os.path.join(metrics_dir, "test.jsonl")

    # Tracker (for metric helpers)
    tracker_experiment = experiment_name or metrics_subdir
    tracker = MetricsTracker(experiment_name=tracker_experiment)

    # Optional resources for span evaluation
    pmids_by_index: List[str] = []
    abstracts_test: Dict[str, str] = {}
    spans_test: Dict[str, List[Dict]] = {}
    if enable_span_eval:
        for pmid, ex_list in test_data.items():
            for _ in ex_list:
                pmids_by_index.append(pmid)
        if not (os.path.exists(md_test.abstract_dict_filepath) and os.path.exists(md_test.spans_dict_filepath)):
            mode = "debug" if debug else "full"
            raise FileNotFoundError(
                (
                    f"Span evaluation enabled but required files are missing.\n"
                    f"Missing: '{md_test.abstract_dict_filepath}' or '{md_test.spans_dict_filepath}'.\n"
                    f"Mode: '{mode}'.\n"
                    f"Please run preprocessing first, e.g.:\n"
                    f"  python -m data_processing.mention_detection.run_md_preprocessing_v2 --mode {mode} --split test"
                )
            )
        abstracts_test = safe_json_load(md_test.abstract_dict_filepath)
        spans_test = safe_json_load(md_test.spans_dict_filepath)

    # Inference on test
    all_logits, all_labels, all_attn = [], [], []
    test_loss_sum = 0.0
    test_loss_count = 0
    with torch.no_grad():
        for input_ids, attention_mask, labels, _ in test_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)
            seq_out = outputs.last_hidden_state
            loss, logits = mention_head(seq_out, ner_labels=labels)
            test_loss_sum += float(loss.item())
            test_loss_count += 1
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            all_attn.append(attention_mask.cpu())

    if not all_logits:
        raise RuntimeError("No test batches processed. Check test data and loader configuration.")

    # Token metrics
    cat_logits = torch.cat(all_logits, dim=0)
    cat_labels = torch.cat(all_labels, dim=0)
    cat_attn = torch.cat(all_attn, dim=0)
    md_metrics = tracker.calculate_md_token_f1(cat_logits, cat_labels)
    num_tokens = int((cat_labels != -100).sum().item())

    # Span metrics
    span_metrics_payload: Dict[str, float] = {}
    if enable_span_eval:
        totals = {
            'strict': {'pred': 0, 'gold': 0, 'matched': 0},
            'lenient': {'pred': 0, 'gold': 0, 'matched': 0},
        }
        by_len = {
            'strict': {'1': {'pred': 0, 'gold': 0, 'matched': 0}, '2_3': {'pred': 0, 'gold': 0, 'matched': 0}, '4plus': {'pred': 0, 'gold': 0, 'matched': 0}},
            'lenient': {'1': {'pred': 0, 'gold': 0, 'matched': 0}, '2_3': {'pred': 0, 'gold': 0, 'matched': 0}, '4plus': {'pred': 0, 'gold': 0, 'matched': 0}},
        }
        by_tail = {
            'strict': {'near_tail': {'pred': 0, 'gold': 0, 'matched': 0}, 'not_near_tail': {'pred': 0, 'gold': 0, 'matched': 0}},
            'lenient': {'near_tail': {'pred': 0, 'gold': 0, 'matched': 0}, 'not_near_tail': {'pred': 0, 'gold': 0, 'matched': 0}},
        }
        dropped_by_truncation = 0

        def bucket_len(span: Tuple[int, int]) -> str:
            ln = int(span[1] - span[0])
            if ln <= 1:
                return '1'
            if ln <= 3:
                return '2_3'
            return '4plus'

        def is_near_tail(span: Tuple[int, int], t_eff: int) -> str:
            tail_start = max(0, int(t_eff) - int(tail_window))
            return 'near_tail' if int(span[1]) >= tail_start else 'not_near_tail'

        total_rows = cat_labels.size(0)
        if len(pmids_by_index) != total_rows:
            raise RuntimeError(
                f"Internal error: pmid index mapping size mismatch (pmids={len(pmids_by_index)} vs rows={total_rows})"
            )

        preds_all = cat_logits.argmax(dim=-1)
        for i in range(total_rows):
            pmid = pmids_by_index[i]
            labels_i = cat_labels[i].tolist()
            preds_i = preds_all[i].tolist()
            attn_i = cat_attn[i]
            t_eff = int(attn_i.sum().item())
            valid_mask = [(lab != -100) for lab in labels_i]

            gold_spans_tok = decode_bio_to_token_spans(labels_i, valid_mask, ignore_index=-100)
            pred_spans_tok = decode_bio_to_token_spans(preds_i, valid_mask, ignore_index=-100)

            totals['strict']['pred'] += len(pred_spans_tok)
            totals['strict']['gold'] += len(gold_spans_tok)
            totals['lenient']['pred'] += len(pred_spans_tok)
            totals['lenient']['gold'] += len(gold_spans_tok)

            m_strict, _, _ = match_spans(pred_spans_tok, gold_spans_tok, mode='strict', iou_threshold=iou_threshold)
            totals['strict']['matched'] += m_strict
            lm = 'any_overlap' if lenient_mode == 'any_overlap' else 'iou'
            m_len, _, _ = match_spans(pred_spans_tok, gold_spans_tok, mode=lm, iou_threshold=iou_threshold)
            totals['lenient']['matched'] += m_len

            for bucket in ('1', '2_3', '4plus'):
                preds_b = [s for s in pred_spans_tok if bucket_len(s) == bucket]
                gold_b = [s for s in gold_spans_tok if bucket_len(s) == bucket]
                by_len['strict'][bucket]['pred'] += len(preds_b)
                by_len['strict'][bucket]['gold'] += len(gold_b)
                by_len['lenient'][bucket]['pred'] += len(preds_b)
                by_len['lenient'][bucket]['gold'] += len(gold_b)
                ms, _, _ = match_spans(preds_b, gold_b, mode='strict', iou_threshold=iou_threshold)
                ml, _, _ = match_spans(preds_b, gold_b, mode=lm, iou_threshold=iou_threshold)
                by_len['strict'][bucket]['matched'] += ms
                by_len['lenient'][bucket]['matched'] += ml

            preds_tail = {'near_tail': [], 'not_near_tail': []}
            gold_tail = {'near_tail': [], 'not_near_tail': []}
            for s in pred_spans_tok:
                preds_tail[is_near_tail(s, t_eff)].append(s)
            for s in gold_spans_tok:
                gold_tail[is_near_tail(s, t_eff)].append(s)
            for region in ('near_tail', 'not_near_tail'):
                pb = preds_tail[region]; gb = gold_tail[region]
                by_tail['strict'][region]['pred'] += len(pb)
                by_tail['strict'][region]['gold'] += len(gb)
                by_tail['lenient'][region]['pred'] += len(pb)
                by_tail['lenient'][region]['gold'] += len(gb)
                ms, _, _ = match_spans(pb, gb, mode='strict', iou_threshold=iou_threshold)
                ml, _, _ = match_spans(pb, gb, mode=lm, iou_threshold=iou_threshold)
                by_tail['strict'][region]['matched'] += ms
                by_tail['lenient'][region]['matched'] += ml

            # Truncation diagnostic via char offsets
            if pmid not in abstracts_test or pmid not in spans_test:
                raise KeyError(f"Missing pmid '{pmid}' in abstracts/spans for span evaluation")
            text = abstracts_test[pmid]
            enc = tokenizer(
                text,
                return_offsets_mapping=True,
                truncation=True,
                max_length=t_eff,
            )
            offsets = enc.get('offset_mapping', [])
            if not offsets or len(offsets) != t_eff:
                raise RuntimeError(
                    f"Offset mapping length mismatch for pmid={pmid}: expected {t_eff}, got {len(offsets) if offsets else 0}"
                )
            gold_char_spans = [(int(s['start']), int(s['end'])) for s in spans_test.get(pmid, [])]
            for (gs, ge) in gold_char_spans:
                has_overlap = False
                for (ts, te) in offsets:
                    if max(ts, gs) < min(te, ge):
                        has_overlap = True
                        break
                if not has_overlap:
                    dropped_by_truncation += 1

        strict_overall = prf(totals['strict']['matched'], totals['strict']['pred'], totals['strict']['gold'])
        lenient_overall = prf(totals['lenient']['matched'], totals['lenient']['pred'], totals['lenient']['gold'])
        span_metrics_payload = {
            'md_span_strict_precision': strict_overall['precision'],
            'md_span_strict_recall': strict_overall['recall'],
            'md_span_strict_f1': strict_overall['f1'],
            'md_span_lenient_precision': lenient_overall['precision'],
            'md_span_lenient_recall': lenient_overall['recall'],
            'md_span_lenient_f1': lenient_overall['f1'],
            'md_span_dropped_by_truncation': float(dropped_by_truncation),
        }
        for regime in ('strict', 'lenient'):
            for bucket in ('1', '2_3', '4plus'):
                stats = by_len[regime][bucket]
                vals = prf(stats['matched'], stats['pred'], stats['gold'])
                span_metrics_payload[f'md_span_{regime}_by_len_{bucket}_precision'] = vals['precision']
                span_metrics_payload[f'md_span_{regime}_by_len_{bucket}_recall'] = vals['recall']
                span_metrics_payload[f'md_span_{regime}_by_len_{bucket}_f1'] = vals['f1']
        for regime in ('strict', 'lenient'):
            for region in ('near_tail', 'not_near_tail'):
                stats = by_tail[regime][region]
                vals = prf(stats['matched'], stats['pred'], stats['gold'])
                span_metrics_payload[f'md_span_{regime}_by_tail_{region}_precision'] = vals['precision']
                span_metrics_payload[f'md_span_{regime}_by_tail_{region}_recall'] = vals['recall']
                span_metrics_payload[f'md_span_{regime}_by_tail_{region}_f1'] = vals['f1']

    # Persist metrics
    record = {
        'split': 'test',
        'md_token_f1': float(md_metrics.get('md_token_f1', md_metrics.get('f1', 0.0))),
        'md_token_precision': float(md_metrics.get('md_token_precision', md_metrics.get('precision', 0.0))),
        'md_token_recall': float(md_metrics.get('md_token_recall', md_metrics.get('recall', 0.0))),
        'num_tokens': int(num_tokens),
        'mean_md_loss': float(test_loss_sum / max(1, test_loss_count)),
    }
    if enable_span_eval:
        record.update({k: float(v) for k, v in span_metrics_payload.items()})

    with open(test_metrics_path, 'a', encoding='utf-8') as f:
        json.dump(record, f); f.write('\n')

    # Console summary
    print(f"Test MD token F1: {record['md_token_f1']:.3f} | P/R: {record['md_token_precision']:.3f}/{record['md_token_recall']:.3f}")
    if enable_span_eval:
        print(
            f"Span(strict) P/R/F1: {span_metrics_payload['md_span_strict_precision']:.3f}/"
            f"{span_metrics_payload['md_span_strict_recall']:.3f}/"
            f"{span_metrics_payload['md_span_strict_f1']:.3f} | "
            f"Span(lenient {lenient_mode}) P/R/F1: {span_metrics_payload['md_span_lenient_precision']:.3f}/"
            f"{span_metrics_payload['md_span_lenient_recall']:.3f}/"
            f"{span_metrics_payload['md_span_lenient_f1']:.3f}"
        )

    return record


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default=None, help='Path to MD checkpoint. If omitted, uses saved_models/<subdir>/best_<debug|full>.pt')
    parser.add_argument('--save_models_dir_name', type=str, default=None, help='Subfolder under saved_models/ where best symlink lives (default: md)')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--experiment_name', type=str, default=None)
    parser.add_argument('--metrics_dir_name', type=str, default=None)
    # Span-eval flags (default ON for testing)
    parser.add_argument('--enable_span_eval', action='store_true', default=True, help='Enable span-level evaluation and logging (default: on)')
    parser.add_argument('--lenient_mode', type=str, default='any_overlap', choices=['any_overlap', 'iou'], help='Lenient span matching mode')
    parser.add_argument('--iou_threshold', type=float, default=0.5, help='IoU threshold when lenient_mode=iou')
    parser.add_argument('--tail_window', type=int, default=64, help='Tail-window size (tokens) for stratified diagnostics')
    args = parser.parse_args()
    main(
        ckpt_path=args.ckpt_path,
        save_models_dir_name=args.save_models_dir_name,
        debug=args.debug,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        experiment_name=args.experiment_name,
        metrics_dir_name=args.metrics_dir_name,
        enable_span_eval=bool(args.enable_span_eval),
        lenient_mode=str(args.lenient_mode),
        iou_threshold=float(args.iou_threshold),
        tail_window=int(args.tail_window),
    )