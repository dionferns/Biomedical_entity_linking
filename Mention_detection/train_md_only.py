"""
Stage A: Mention Detection (MD-only) training script.

Trains the encoder + MentionDetection head on BIO token labels using the
existing preprocessed MedMention tensors. Saves best checkpoint by MD token F1.
"""

import os
import json
from datetime import datetime
from typing import Optional, Dict, List, Tuple

import torch
from torch.optim import AdamW
from transformers import BertModel, BertTokenizerFast
import sys
from pathlib import Path

# Ensure project root is on sys.path so intra-project imports work when running as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_processing.mention_detection.md_preprocessing_v2 import MentionDetectionPreprocessingV2
from models.mention_detection import MentionDetection
from utils.metrics_tracker import MetricsTracker
from utils.error_handling import safe_json_load
from utils.md_span_metrics import decode_bio_to_token_spans, match_spans, prf


def _ensure_split_ready(debug: bool, split: str, tokenizer: BertTokenizerFast):
    md = MentionDetectionPreprocessingV2(debug=debug, split_type=split)
    # Fail-fast: this training script must not create any preprocessing artifacts.
    # It assumes all required files already exist (produced by the preprocessing pipeline).
    required_path = md.tokenized_filepath
    if not os.path.exists(required_path):
        mode = "debug" if debug else "full"
        raise FileNotFoundError(
            (
                f"Missing required tokenized tensors for mention detection: {required_path}\n"
                f"Split: '{split}', Mode: '{mode}'.\n"
                f"Please run preprocessing first, e.g.:\n"
                f"  python -m data_processing.mention_detection.run_md_preprocessing_v2 --mode {mode} --split {split}"
            )
        )
    data = torch.load(required_path)
    return data


def main(
    debug: bool = True,
    num_epochs: int = 5,
    batch_size: int = 8,
    num_workers: Optional[int] = 4,
    learning_rate: float = 5e-5,
    experiment_name: Optional[str] = None,
    save_models_dir_name: Optional[str] = None,
    metrics_dir_name: Optional[str] = None,
    enable_span_eval: bool = False,
    lenient_mode: str = 'any_overlap',
    iou_threshold: float = 0.5,
    tail_window: int = 64,
    use_span_strict_for_selection: bool = False,
):
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

    # Data
    train_data = _ensure_split_ready(debug, "train", tokenizer)
    val_data = _ensure_split_ready(debug, "val", tokenizer)

    train_loader = MentionDetectionPreprocessingV2.create_dataloader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = MentionDetectionPreprocessingV2.create_dataloader(
        val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    optimizer = AdamW(list(encoder.parameters()) + list(mention_head.parameters()), lr=learning_rate)
    # Align tracker experiment folder with save_models_dir_name when provided
    tracker_experiment = experiment_name or save_models_dir_name or f"md_only_{debug_suffix}"
    tracker = MetricsTracker(experiment_name=tracker_experiment)

    # Resolve checkpoint directory from user input or default
    save_subdir = save_models_dir_name.strip() if save_models_dir_name else "md"
    save_dir = os.path.join("saved_models", save_subdir)
    os.makedirs(save_dir, exist_ok=True)
    run_tag = f"{debug_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    best_f1 = -1.0
    best_ckpt = None

    def _ckpt_path(epoch: int) -> str:
        return os.path.join(save_dir, f"md_checkpoint_epoch_{epoch}_{run_tag}.pt")

    # Prepare metrics output directory and files
    metrics_subdir = metrics_dir_name.strip() if metrics_dir_name else f"md_{debug_suffix}"
    metrics_dir = os.path.join("results", metrics_subdir)
    os.makedirs(metrics_dir, exist_ok=True)
    train_metrics_path = os.path.join(metrics_dir, "train.jsonl")
    val_metrics_path = os.path.join(metrics_dir, "val.jsonl")

    # Redirect tracker artifact files (metrics.json, summary.txt, latest_metrics.json) into results when metrics_dir_name is provided
    try:
        from pathlib import Path
        if metrics_dir_name:
            tracker.save_dir = Path(metrics_dir)
            tracker.metrics_file = tracker.save_dir / "metrics.json"
            tracker.summary_file = tracker.save_dir / "metrics_summary.txt"
            # Also override base for latest files to avoid writing under saved_models
            tracker.config.checkpoint_dir = 'results'
    except Exception:
        pass

    # Prepare optional resources for span evaluation
    pmids_by_index: List[str] = []
    abstracts_val: Dict[str, str] = {}
    spans_val: Dict[str, List[Dict]] = {}
    if enable_span_eval:
        # Build a deterministic mapping from dataset row index -> pmid
        for pmid, ex_list in val_data.items():
            for _ in ex_list:
                pmids_by_index.append(pmid)
        # Load required JSONs and fail fast if missing
        md_val = MentionDetectionPreprocessingV2(debug=debug, split_type="val")
        if not (os.path.exists(md_val.abstract_dict_filepath) and os.path.exists(md_val.spans_dict_filepath)):
            mode = "debug" if debug else "full"
            raise FileNotFoundError(
                (
                    f"Span evaluation enabled but required files are missing.\n"
                    f"Missing: '{md_val.abstract_dict_filepath}' or '{md_val.spans_dict_filepath}'.\n"
                    f"Split: 'val', Mode: '{mode}'.\n"
                    f"Please run preprocessing first, e.g.:\n"
                    f"  python -m data_processing.mention_detection.run_md_preprocessing_v2 --mode {mode} --split val"
                )
            )
        abstracts_val = safe_json_load(md_val.abstract_dict_filepath)
        spans_val = safe_json_load(md_val.spans_dict_filepath)

    last_val_md_metrics = {'f1': 0.0}
    for epoch in range(1, num_epochs + 1):
        # Train
        encoder.train(); mention_head.train()
        tracker.start_epoch(epoch, 'train')
        train_loss_sum = 0.0
        train_loss_count = 0
        for input_ids, attention_mask, labels, _ in train_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)
            seq_out = outputs.last_hidden_state
            loss, logits = mention_head(seq_out, ner_labels=labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(mention_head.parameters()), max_norm=1.0)
            optimizer.step()
            loss_val = float(loss.item())
            tracker.update_batch_metrics('train', batch_size=input_ids.size(0), losses={'md_loss': loss_val})
            if loss_val == loss_val and loss_val not in (float('inf'), float('-inf')):
                train_loss_sum += loss_val
                train_loss_count += 1
        tracker.end_epoch('train')
        # Persist train metrics (mean md_loss) for this epoch
        try:
            mean_train_loss = (train_loss_sum / max(1, train_loss_count)) if train_loss_count > 0 else None
            with open(train_metrics_path, 'a', encoding='utf-8') as f_tm:
                json.dump({'epoch': int(epoch), 'mean_md_loss': None if mean_train_loss is None else float(mean_train_loss)}, f_tm)
                f_tm.write('\n')
        except Exception:
            pass

        # Validate
        encoder.eval(); mention_head.eval()
        tracker.start_epoch(epoch, 'val')
        all_logits, all_labels, all_attn = [], [], []
        with torch.no_grad():
            row_offset = 0
            for input_ids, attention_mask, labels, _ in val_loader:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)
                outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)
                seq_out = outputs.last_hidden_state
                loss, logits = mention_head(seq_out, ner_labels=labels)
                cpu_logits = logits.cpu()
                cpu_labels = labels.cpu()
                all_logits.append(cpu_logits)
                all_labels.append(cpu_labels)
                all_attn.append(attention_mask.cpu())
                tracker.update_batch_metrics('val', batch_size=input_ids.size(0), losses={'md_loss': float(loss.item())})
        # Metrics
        md_metrics = {'f1': 0.0, 'precision': 0.0, 'recall': 0.0}
        num_tokens = 0
        span_metrics_payload: Dict[str, float] = {}
        if all_logits:
            cat_logits = torch.cat(all_logits, dim=0)
            cat_labels = torch.cat(all_labels, dim=0)
            cat_attn = torch.cat(all_attn, dim=0)
            md_metrics = tracker.calculate_md_token_f1(cat_logits, cat_labels)
            # Count evaluated tokens (exclude ignore_index=-100)
            num_tokens = int((cat_labels != -100).sum().item())

            if enable_span_eval:
                # Prepare accumulators
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

                # Iterate each sequence/sample
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

                    # Totals
                    totals['strict']['pred'] += len(pred_spans_tok)
                    totals['strict']['gold'] += len(gold_spans_tok)
                    totals['lenient']['pred'] += len(pred_spans_tok)
                    totals['lenient']['gold'] += len(gold_spans_tok)

                    # Match strict
                    m_strict, _, _ = match_spans(pred_spans_tok, gold_spans_tok, mode='strict', iou_threshold=iou_threshold)
                    totals['strict']['matched'] += m_strict
                    # Match lenient
                    lm = 'any_overlap' if lenient_mode == 'any_overlap' else 'iou'
                    m_len, _, _ = match_spans(pred_spans_tok, gold_spans_tok, mode=lm, iou_threshold=iou_threshold)
                    totals['lenient']['matched'] += m_len

                    # Stratify by length buckets
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

                    # Stratify by tail window
                    preds_tail = {'near_tail': [], 'not_near_tail': []}
                    gold_tail = {'near_tail': [], 'not_near_tail': []}
                    for s in pred_spans_tok:
                        preds_tail[is_near_tail(s, t_eff)].append(s)
                    for s in gold_spans_tok:
                        gold_tail[is_near_tail(s, t_eff)].append(s)
                    for region in ('near_tail', 'not_near_tail'):
                        pb = preds_tail[region]
                        gb = gold_tail[region]
                        by_tail['strict'][region]['pred'] += len(pb)
                        by_tail['strict'][region]['gold'] += len(gb)
                        by_tail['lenient'][region]['pred'] += len(pb)
                        by_tail['lenient'][region]['gold'] += len(gb)
                        ms, _, _ = match_spans(pb, gb, mode='strict', iou_threshold=iou_threshold)
                        ml, _, _ = match_spans(pb, gb, mode=lm, iou_threshold=iou_threshold)
                        by_tail['strict'][region]['matched'] += ms
                        by_tail['lenient'][region]['matched'] += ml

                    # Count dropped_by_truncation via char offsets (re-tokenize up to t_eff)
                    if pmid not in abstracts_val or pmid not in spans_val:
                        raise KeyError(f"Missing pmid '{pmid}' in abstracts/spans for span evaluation")
                    text = abstracts_val[pmid]
                    enc = tokenizer(
                        text,
                        return_offsets_mapping=True,
                        truncation=True,
                        max_length=t_eff,
                    )
                    offsets = enc.get('offset_mapping', [])
                    # offsets length may be < t_eff if tokenizer collapses specials differently; validate
                    if not offsets or len(offsets) != t_eff:
                        raise RuntimeError(
                            f"Offset mapping length mismatch for pmid={pmid}: expected {t_eff}, got {len(offsets) if offsets else 0}"
                        )
                    # Build a quick check: any token that overlaps a char span within included window
                    gold_char_spans = [(int(s['start']), int(s['end'])) for s in spans_val.get(pmid, [])]
                    for (gs, ge) in gold_char_spans:
                        has_overlap = False
                        for (ts, te) in offsets:
                            if max(ts, gs) < min(te, ge):
                                has_overlap = True
                                break
                        if not has_overlap:
                            dropped_by_truncation += 1

                # Compose flat metrics payload
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
                # by length
                for regime in ('strict', 'lenient'):
                    for bucket in ('1', '2_3', '4plus'):
                        stats = by_len[regime][bucket]
                        vals = prf(stats['matched'], stats['pred'], stats['gold'])
                        span_metrics_payload[f'md_span_{regime}_by_len_{bucket}_precision'] = vals['precision']
                        span_metrics_payload[f'md_span_{regime}_by_len_{bucket}_recall'] = vals['recall']
                        span_metrics_payload[f'md_span_{regime}_by_len_{bucket}_f1'] = vals['f1']
                # by tail
                for regime in ('strict', 'lenient'):
                    for region in ('near_tail', 'not_near_tail'):
                        stats = by_tail[regime][region]
                        vals = prf(stats['matched'], stats['pred'], stats['gold'])
                        span_metrics_payload[f'md_span_{regime}_by_tail_{region}_precision'] = vals['precision']
                        span_metrics_payload[f'md_span_{regime}_by_tail_{region}_recall'] = vals['recall']
                        span_metrics_payload[f'md_span_{regime}_by_tail_{region}_f1'] = vals['f1']

                # Concise console log
                try:
                    print(
                        f"Span(strict) P/R/F1: {strict_overall['precision']:.3f}/{strict_overall['recall']:.3f}/{strict_overall['f1']:.3f} | "
                        f"Span(lenient) P/R/F1: {lenient_overall['precision']:.3f}/{lenient_overall['recall']:.3f}/{lenient_overall['f1']:.3f} | "
                        f"Dropped by truncation: {int(dropped_by_truncation)}"
                    )
                except Exception:
                    pass
        last_val_md_metrics = md_metrics
        tracker.end_epoch('val')
        tracker.update_epoch_metrics('val', md_metrics=md_metrics)
        tracker.save_metrics()
        # Persist val metrics for this epoch
        try:
            with open(val_metrics_path, 'a', encoding='utf-8') as f_vm:
                rec = {
                    'epoch': int(epoch),
                    'md_token_f1': float(md_metrics.get('md_token_f1', md_metrics.get('f1', 0.0))),
                    'md_token_precision': float(md_metrics.get('md_token_precision', md_metrics.get('precision', 0.0))),
                    'md_token_recall': float(md_metrics.get('md_token_recall', md_metrics.get('recall', 0.0))),
                    'num_tokens': int(num_tokens),
                }
                if enable_span_eval and span_metrics_payload:
                    rec.update({k: float(v) for k, v in span_metrics_payload.items()})
                json.dump(rec, f_vm); f_vm.write('\n')
        except Exception:
            pass

        # Save checkpoint
        ckpt_path = _ckpt_path(epoch)
        torch.save({
            'epoch': epoch,
            'encoder_state_dict': encoder.state_dict(),
            'mention_head_state_dict': mention_head.state_dict(),
        }, ckpt_path)
        print(f"Saved MD checkpoint: {ckpt_path}")

        if enable_span_eval and use_span_strict_for_selection and span_metrics_payload:
            f1_val = float(span_metrics_payload.get('md_span_strict_f1', 0.0))
        else:
            f1_val = float(md_metrics.get('f1') or md_metrics.get('md_token_f1') or 0.0)
        if f1_val >= best_f1:
            best_f1 = f1_val
            best_ckpt = ckpt_path

    # Write symlink/file with best path
    if best_ckpt:
        best_link = os.path.join(save_dir, f"best_{debug_suffix}.pt")
        try:
            if os.path.islink(best_link) or os.path.exists(best_link):
                os.remove(best_link)
            os.symlink(os.path.abspath(best_ckpt), best_link)
        except Exception:
            # Fallback: copy lightweight marker JSON
            with open(os.path.join(save_dir, f"best_{debug_suffix}.json"), 'w') as f:
                json.dump({'best': best_ckpt}, f)
        print(f"Best MD checkpoint: {best_ckpt}")
    # Save concise summary under metrics directory
    try:
        summary = {
            'stage': 'md',
            'debug': bool(debug),
            'best_checkpoint': best_ckpt,
            'val_md_f1': float(last_val_md_metrics.get('f1') or last_val_md_metrics.get('md_token_f1') or 0.0),
            'val_md_precision': float(last_val_md_metrics.get('precision') or last_val_md_metrics.get('md_token_precision') or 0.0),
            'val_md_recall': float(last_val_md_metrics.get('recall') or last_val_md_metrics.get('md_token_recall') or 0.0),
        }
        with open(os.path.join(metrics_dir, 'summary.json'), 'w', encoding='utf-8') as fsum:
            json.dump(summary, fsum, indent=2)
    except Exception:
        pass
    return best_ckpt or ""


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--experiment_name', type=str, default=None)
    parser.add_argument('--save_models_dir_name', type=str, default=None, help='Subfolder under saved_models/ to store checkpoints')
    parser.add_argument('--metrics_dir_name', type=str, default=None, help='Subfolder under results/ to store per-epoch metrics')
    # Span-eval related flags (all optional; default off for backward-compat)
    parser.add_argument('--enable_span_eval', action='store_true', help='Enable span-level evaluation and logging')
    parser.add_argument('--lenient_mode', type=str, default='any_overlap', choices=['any_overlap', 'iou'], help='Lenient span matching mode')
    parser.add_argument('--iou_threshold', type=float, default=0.5, help='IoU threshold when lenient_mode=iou')
    parser.add_argument('--tail_window', type=int, default=64, help='Tail-window size (tokens) for stratified diagnostics')
    parser.add_argument('--use_span_strict_for_selection', action='store_true', help='Use strict span F1 for best-checkpoint selection')
    args = parser.parse_args()
    main(
        debug=args.debug,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        learning_rate=args.lr,
        experiment_name=args.experiment_name,
        save_models_dir_name=args.save_models_dir_name,
        metrics_dir_name=args.metrics_dir_name,
        enable_span_eval=bool(args.enable_span_eval),
        lenient_mode=str(args.lenient_mode),
        iou_threshold=float(args.iou_threshold),
        tail_window=int(args.tail_window),
        use_span_strict_for_selection=bool(args.use_span_strict_for_selection),
    )



# 7000 mid
# python -m train.train_md_only --epochs 6 --batch_size 32 --num_workers 12 --lr 5e-5   --experiment_name md_only_full_span --metrics_dir_name md_only_full_span   --enable_span_eval   --lenient_mode any_overlap   --iou_threshold 0.5   --tail_window 64