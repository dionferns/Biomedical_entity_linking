"""
Build mention-centered context strings for Cross-Encoder training.

Reads validation abstract text and gold spans, then creates a JSON mapping:

  mention_id -> {
    "sentence": <substring around mention of fixed char window>,
    "mention": <gold mention text>,
    "cui": <gold CUI id>
  }

Notes:
- Uses abstract IDs only to locate mentions; abstract IDs are not included in output.
- Fails fast with clear errors if inputs are missing or span indices are invalid.
- The "sentence" field contains the mention marked with [M_START] and [M_END].

Example:
  python cross_encoder/build_ce_contexts.py \
    --abstract_path data/processed/mention_detection/val/abstract_dict_full.json \
    --spans_path data/processed/mention_detection/val/spans_dict_full.json \
    --output_path cross_encoder/ce_val_mention_sentences.json \
    --window 256
"""

import argparse
import os
import json
import gzip
from typing import Dict, Any

from utils.error_handling import safe_json_load


def extract_window(text: str, start: int, end: int, window: int) -> str:
    """
    Extract a fixed-width character window around a span [start, end) from text.

    The resulting substring starts at max(0, start - window//2) and ends at
    min(len(text), end + window//2).

    Args:
        text: Full abstract text
        start: Mention start char index (inclusive)
        end: Mention end char index (exclusive)
        window: Total number of context characters around the mention

    Returns:
        Substring including the mention and surrounding context
    """
    if start < 0 or end < 0 or start > end or end > len(text):
        raise ValueError(
            f"Invalid span indices: start={start}, end={end}, text_len={len(text)}"
        )
    half = max(0, int(window) // 2)
    left = max(0, start - half)
    right = min(len(text), end + half)
    return text[left:right]


def build_marked_window(text: str, start: int, end: int, window: int) -> str:
    """
    Extract a window around the span and insert [M_START] and [M_END] around the gold mention.

    The marker positions are computed from the provided span indices, not string search.

    Args:
        text: Full abstract text
        start: Mention start char index (inclusive)
        end: Mention end char index (exclusive)
        window: Total number of context characters around the mention

    Returns:
        Window substring with [M_START]mention[M_END] inserted
    """
    if start < 0 or end < 0 or start > end or end > len(text):
        raise ValueError(
            f"Invalid span indices: start={start}, end={end}, text_len={len(text)}"
        )
    half = max(0, int(window) // 2)
    left = max(0, start - half)
    right = min(len(text), end + half)
    # Relative indices of mention inside the window
    rel_start = start - left
    rel_end = end - left
    window_text = text[left:right]
    return (
        window_text[:rel_start]
        + "[M_START] "
        + window_text[rel_start:rel_end]
        + " [M_END]"
        + window_text[rel_end:]
    )


def build_mention_contexts(
    abstract_path: str,
    spans_path: str,
    window: int,
) -> Dict[str, Dict[str, Any]]:
    """
    Build mapping from mention_id -> {"sentence": mention-centered substring}.

    Args:
        abstract_path: Path to abstract_dict_*.json
        spans_path: Path to spans_dict_*.json
        window: Character window size around mention

    Returns:
        Dictionary keyed by mention_id (as string)
    """
    if not os.path.exists(abstract_path):
        raise FileNotFoundError(f"Abstract file not found: {abstract_path}")
    if not os.path.exists(spans_path):
        raise FileNotFoundError(f"Spans file not found: {spans_path}")

    abstracts = safe_json_load(abstract_path)  # {abstract_id: text}
    spans = safe_json_load(spans_path)        # {abstract_id: [ {id, start, end, mention, ...}, ... ]}

    out: Dict[str, Dict[str, Any]] = {}
    seen_ids = set()

    for abs_id, span_list in spans.items():
        if abs_id not in abstracts:
            raise KeyError(
                f"Abstract id {abs_id} present in spans but missing in abstracts file"
            )
        text = abstracts[abs_id]
        if not isinstance(text, str) or not text:
            raise ValueError(f"Empty or invalid abstract text for id {abs_id}")

        if not isinstance(span_list, list):
            raise ValueError(f"Spans entry for abstract id {abs_id} must be a list")

        for rec in span_list:
            if not isinstance(rec, dict):
                raise ValueError(f"Span record must be a dict for abstract id {abs_id}")
            if (
                "id" not in rec
                or "start" not in rec
                or "end" not in rec
                or "mention" not in rec
                or "cui" not in rec
            ):
                raise KeyError(
                    f"Span record missing required keys (id/start/end/mention/cui) for abstract id {abs_id}"
                )

            mention_id_str = str(rec["id"])  # normalize to string key
            if mention_id_str in seen_ids:
                raise RuntimeError(f"Duplicate mention id encountered: {mention_id_str}")

            start = int(rec["start"])  # expected char offsets
            end = int(rec["end"])      # exclusive
            mention_text = rec.get("mention", "")
            cui = rec.get("cui")

            # Validate span alignment
            if start < 0 or end > len(text) or start >= end:
                raise ValueError(
                    f"Invalid span for mention {mention_id_str}: start={start}, end={end}, text_len={len(text)}"
                )
            if text[start:end] != mention_text:
                raise ValueError(
                    f"Span text mismatch for mention {mention_id_str}: '\{text[start:end]}' != '\{mention_text}'"
                )

            ctx = build_marked_window(text, start, end, window)
            out[mention_id_str] = {"sentence": ctx, "mention": mention_text, "cui": cui}
            seen_ids.add(mention_id_str)

    if not out:
        raise RuntimeError("No mentions were processed; output is empty")

    return out


def main():
    parser = argparse.ArgumentParser(
        description="Build mention-centered context strings for Cross-Encoder training."
    )
    parser.add_argument(
        "--abstract_path",
        type=str,
        default=os.path.join(
            "data", "processed", "mention_detection", "val", "abstract_dict_full.json"
        ),
        help="Path to abstract_dict_*.json",
    )
    parser.add_argument(
        "--spans_path",
        type=str,
        default=os.path.join(
            "data", "processed", "mention_detection", "val", "spans_dict_full.json"
        ),
        help="Path to spans_dict_*.json",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=os.path.join("cross_encoder", "ce_val_mention_sentences.jsonl.gz"),
        help="Where to save output mapping (JSONL; one JSON per line). If ends with .gz, writes gzipped.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=256,
        help="Character window size around the mention span",
    )
    args = parser.parse_args()

    result = build_mention_contexts(args.abstract_path, args.spans_path, args.window)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    is_gz = args.output_path.endswith(".gz")
    opener = gzip.open if is_gz else open
    # Write JSONL, preserving iteration order from the result dict
    with opener(args.output_path, "wt", encoding="utf-8") as f:
        for mention_id, payload in result.items():
            rec = dict(payload)
            rec["mention_id"] = str(mention_id)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(
        f"Saved {len(result)} mention contexts to {args.output_path} (window={args.window})"
    )


if __name__ == "__main__":
    main()


