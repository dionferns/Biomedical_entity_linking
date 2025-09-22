#!/usr/bin/env python3
"""
Combine CE eval candidates and contexts into a single JSONL.GZ file.

Inputs (JSONL.GZ):
- eval_jsonl_gz: Output from eval_faiss_candidates_dump.py
  Each line has: {mention_id, mention, gold_cui, candidates: [ {cui, name, is_gold, is_null, faiss_rank} ... ]}

- contexts_jsonl_gz: Output from build_ce_contexts.py
  Each line has: {mention_id, sentence, mention, cui}

Output (JSONL.GZ): One line per retrieved candidate for each joined mention_id:
{
  "group_id": "val_debug1",
  "mention_id": "1",
  "mention_cui": "C0027051",
  "is_gold": true,
  "is_nil": false,
  "context_text": "... [M_START] ... [M_END] ...",
  "cand_text": "myocardial infarction",
  "cand_cui": "C0027051",
  "faiss_rank": 1
}

Notes:
- group_id is formed as {group_prefix}{seq}, where seq starts at --start_index (default 1).
- faiss_rank is emitted as an integer (converted from input string).
- is_nil is mapped from the eval candidate's is_null.
- Mentions present in only one input are printed to stderr and skipped.
- If --mrdef_path is provided (defaults to cross_encoder/MRDEF_unique_definitions.RRF), then for each candidate
  we add "cand_def" using the candidate's CUI matched to MRDEF. The value is the first 100 characters of the
  definition text. For NIL candidates, "cand_def" is "[NIL]"; for non-NIL candidates with no definition, "cand_def" is "[NO_DEF]".
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import sys
from typing import Dict, Any, Iterable, Iterator, List, Optional, Tuple


def iter_jsonl_gz(path: str) -> Iterator[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input not found: {path}")
    with gzip.open(path, mode="rt", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            yield json.loads(line)


def load_contexts(path: str) -> Dict[str, Dict[str, Any]]:
    contexts: Dict[str, Dict[str, Any]] = {}
    for rec in iter_jsonl_gz(path):
        mention_id = str(rec.get("mention_id", "")).strip()
        if not mention_id:
            continue
        contexts[mention_id] = {
            "context_text": rec.get("sentence", ""),
            "mention_cui": rec.get("cui"),
        }
    return contexts


def safe_int(value: Any, default: int = -1) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        try:
            return int(str(value))
        except Exception:
            return default


def open_text_maybe_gzip(path: str) -> io.TextIOBase:
    if path.endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, mode="rb"), encoding="utf-8", errors="replace")
    return open(path, mode="r", encoding="utf-8", errors="replace")


def load_mrdef_definitions(mrdef_path: Optional[str]) -> Dict[str, str]:
    """Load CUI -> definition map from MRDEF.

    Expects RRF columns: 0:CUI | 5:DEF (minimum). Returns first definition per CUI.
    Returns empty map if mrdef_path is falsy.
    """
    defs: Dict[str, str] = {}
    if not mrdef_path:
        return defs
    if not os.path.exists(mrdef_path):
        # Silent fallback to empty; caller can proceed with [NIL]
        return defs
    with open_text_maybe_gzip(mrdef_path) as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\n")
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 6:
                continue
            cui = parts[0].strip()
            definition_text = parts[5].strip()
            if not cui or not definition_text:
                continue
            # Keep first seen definition
            if cui not in defs:
                defs[cui] = definition_text
    return defs


def truncate_by_words(text: str, max_words: int) -> str:
    """Return the first max_words whitespace-delimited words from text.

    Strips leading/trailing whitespace. max_words must be positive.
    """
    if int(max_words) <= 0:
        raise ValueError(f"max_words must be positive, got {max_words}")
    words = (text or "").strip().split()
    if not words:
        return ""
    return " ".join(words[:int(max_words)])


def combine(
    eval_jsonl_gz: str,
    contexts_jsonl_gz: str,
    out_jsonl_gz: str,
    group_prefix: str,
    start_index: int = 1,
    mrdef_path: Optional[str] = None,
    def_truncation: str = "chars",
    def_max_chars: int = 100,
    def_max_words: int = 20,
) -> Tuple[int, int, int]:
    contexts = load_contexts(contexts_jsonl_gz)
    ctx_ids = set(contexts.keys())

    os.makedirs(os.path.dirname(out_jsonl_gz) or ".", exist_ok=True)

    seq = int(start_index) - 1
    seen_eval_ids: List[str] = []
    written = 0

    defs_map = load_mrdef_definitions(mrdef_path)

    with gzip.open(out_jsonl_gz, mode="wt", encoding="utf-8") as out_f:
        for rec in iter_jsonl_gz(eval_jsonl_gz):
            mention_id = str(rec.get("mention_id", "")).strip()
            if not mention_id:
                continue
            seen_eval_ids.append(mention_id)
            if mention_id not in contexts:
                continue

            seq += 1
            group_id = f"{group_prefix}{seq}"
            ctx = contexts[mention_id]
            mention_cui = ctx.get("mention_cui")
            context_text = ctx.get("context_text", "")

            for cand in rec.get("candidates", []):
                cand_cui = cand.get("cui")
                cand_text = cand.get("name", "")
                is_gold = bool(cand.get("is_gold", False))
                is_nil = bool(cand.get("is_null", False))
                faiss_rank = safe_int(cand.get("faiss_rank"), default=-1)
                # Determine candidate definition (truncate per configuration)
                if is_nil or not cand_cui or cand_cui == "[NIL]":
                    cand_def = "[NIL]"
                else:
                    full_def = defs_map.get(str(cand_cui))
                    if not full_def:
                        cand_def = "[NO_DEF]"
                    else:
                        if def_truncation == "chars":
                            if int(def_max_chars) <= 0:
                                raise ValueError(f"def_max_chars must be positive, got {def_max_chars}")
                            cand_def = full_def[:int(def_max_chars)]
                        elif def_truncation == "words":
                            cand_def = truncate_by_words(full_def, int(def_max_words))
                        else:
                            raise ValueError(f"Invalid def_truncation: {def_truncation}")

                out_rec = {
                    "group_id": group_id,
                    "mention_id": mention_id,
                    "mention_cui": mention_cui,
                    "is_gold": is_gold,
                    "is_nil": is_nil,
                    "context_text": context_text,
                    "cand_text": cand_text,
                    "cand_cui": cand_cui,
                    "faiss_rank": faiss_rank,
                    "cand_def": cand_def,
                }
                out_f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                written += 1

    eval_ids = set(seen_eval_ids)
    only_eval = sorted(eval_ids - ctx_ids, key=lambda x: (len(x), x))
    only_ctx = sorted(ctx_ids - eval_ids, key=lambda x: (len(x), x))
    if only_eval:
        print(f"[warn] {len(only_eval)} mention_id(s) present in eval only:", file=sys.stderr)
        for mid in only_eval:
            print(mid, file=sys.stderr)
    if only_ctx:
        print(f"[warn] {len(only_ctx)} mention_id(s) present in contexts only:", file=sys.stderr)
        for mid in only_ctx:
            print(mid, file=sys.stderr)

    return written, len(only_eval), len(only_ctx)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Combine eval_faiss_candidates_dump and build_ce_contexts outputs into a per-candidate JSONL.GZ."
        )
    )
    parser.add_argument("--eval_jsonl_gz", required=True, help="Path to eval candidates JSONL.GZ")
    parser.add_argument("--contexts_jsonl_gz", required=True, help="Path to contexts JSONL.GZ")
    parser.add_argument(
        "--out_jsonl_gz",
        required=False,
        default=None,
        help="Output path for combined JSONL.GZ (default: cross_encoder/combined_<group_prefix>.jsonl.gz)",
    )
    parser.add_argument(
        "--mrdef_path",
        required=False,
        default=os.path.join("cross_encoder", "MRDEF_unique_definitions.RRF"),
        help="Path to MRDEF definitions file (RRF or RRF.gz) to annotate cand_def (truncate by chars or words).",
    )
    parser.add_argument(
        "--def_truncation",
        required=False,
        choices=["chars", "words"],
        default="chars",
        help="Truncation mode for definitions: 'chars' (default) or 'words'",
    )
    parser.add_argument(
        "--def_max_chars",
        type=int,
        required=False,
        default=100,
        help="Max characters when --def_truncation=chars (default: 100)",
    )
    parser.add_argument(
        "--def_max_words",
        type=int,
        required=False,
        default=20,
        help="Max words when --def_truncation=words (default: 20)",
    )
    parser.add_argument(
        "--group_prefix",
        required=False,
        default="val_debug",
        help="Prefix for group_id, concatenated with a running sequence (e.g., 'val_debug').",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        required=False,
        default=1,
        help="Starting index for group numbering (default: 1)",
    )

    args = parser.parse_args(argv)

    out_path = args.out_jsonl_gz
    if not out_path:
        out_dir = os.path.join("cross_encoder")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"combined_{args.group_prefix}.jsonl.gz")

    # Fail-fast validations for truncation configuration
    if args.def_truncation not in ("chars", "words"):
        raise ValueError(f"--def_truncation must be 'chars' or 'words', got {args.def_truncation}")
    if args.def_truncation == "chars" and int(args.def_max_chars) <= 0:
        raise ValueError(f"--def_max_chars must be positive, got {args.def_max_chars}")
    if args.def_truncation == "words" and int(args.def_max_words) <= 0:
        raise ValueError(f"--def_max_words must be positive, got {args.def_max_words}")

    written, only_eval, only_ctx = combine(
        eval_jsonl_gz=args.eval_jsonl_gz,
        contexts_jsonl_gz=args.contexts_jsonl_gz,
        out_jsonl_gz=out_path,
        group_prefix=str(args.group_prefix),
        start_index=int(args.start_index),
        mrdef_path=str(args.mrdef_path) if args.mrdef_path else None,
        def_truncation=str(args.def_truncation),
        def_max_chars=int(args.def_max_chars),
        def_max_words=int(args.def_max_words),
    )

    print(
        json.dumps(
            {
                "written": int(written),
                "unmatched_eval_only": int(only_eval),
                "unmatched_contexts_only": int(only_ctx),
                "output": out_path,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


