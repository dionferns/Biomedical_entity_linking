#!/usr/bin/env python3
"""
Train a cross-encoder on grouped (context, candidate) pairs produced by combine_ce_eval_and_contexts.py.

Inputs (JSONL.GZ):
- Each line contains: group_id, mention_id, context_text (with [M_START]/[M_END]),
  cand_text, cand_def ("[NO_DEF]" if missing), cand_cui (or "[NIL]"), is_gold (bool),
  is_nil (bool), faiss_rank (int)

Behavior:
- Groups lines by group_id. Validates exactly one is_gold==True per group.
- Sorts candidates within each group by ascending faiss_rank.
- Tokenizes as a single sequence with explicit segment layout:
  [CLS] context [SEP] cand_text [SEP] cand_def [SEP]
  Token type ids: 0 on [CLS] context [SEP], 1 on subsequent tokens.
- Computes group-wise cross-entropy over per-candidate scores.
- Reports grouped top-1 accuracy and average loss on train/eval splits.

Fail-fast policy:
- Missing required fields, malformed groups, or invalid values raise errors immediately.
- No implicit fallbacks. The script stops with actionable error messages.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import random
import sys
import errno
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast


ADDITIONAL_SPECIAL_TOKENS: List[str] = ["[M_START]", "[M_END]", "[NO_DEF]", "[NIL]"]


def iter_jsonl_gz(path: str) -> Iterator[Dict[str, Any]]:
    """Yield JSON objects from a .jsonl.gz file.

    Raises FileNotFoundError if path does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input not found: {path}")
    with gzip.open(path, mode="rt", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                raise ValueError(f"Invalid JSON line: {line[:200]}") from e
            yield obj


@dataclass
class Candidate:
    cand_text: str
    cand_def: str
    cand_cui: Optional[str]
    is_gold: bool
    is_nil: bool
    faiss_rank: int


@dataclass
class Group:
    group_id: str
    mention_id: str
    context_text: str
    candidates: List[Candidate]
    gold_index: int

def _require_field(obj: Dict[str, Any], key: str) -> Any:
    if key not in obj:
        raise KeyError(f"Missing required field '{key}' in record: {obj}")
    return obj[key]

def _freeze_first_n_layers(model, n: int) -> None:
    base = getattr(model, "bert", None) or getattr(model, "roberta", None) or getattr(model, "deberta", None)
    if base is None:
        return
    enc = getattr(base, "encoder", None)
    layers = getattr(enc, "layer", None)
    if not layers:
        return
    n = max(0, min(int(n), len(layers)))
    for i in range(n):
        for p in layers[i].parameters():
            p.requires_grad = False

def load_groups(path: str) -> List[Group]:
    """Load and validate groups from combined JSONL.GZ.

    - Groups by group_id
    - Validates exactly one is_gold per group
    - Sorts candidates by ascending faiss_rank
    """
    groups_map: Dict[str, Dict[str, Any]] = {}
    for rec in iter_jsonl_gz(path):
        group_id = str(_require_field(rec, "group_id"))
        mention_id = str(_require_field(rec, "mention_id"))
        context_text = str(_require_field(rec, "context_text"))
        cand_text = str(_require_field(rec, "cand_text"))
        # cand_def may be missing; acceptable to normalize to "[NO_DEF]" per data assumptions
        cand_def = rec.get("cand_def")
        if not isinstance(cand_def, str) or not cand_def:
            cand_def = "[NO_DEF]"
        cand_cui = rec.get("cand_cui")
        is_gold = bool(_require_field(rec, "is_gold"))
        is_nil = bool(_require_field(rec, "is_nil"))
        faiss_rank_raw = _require_field(rec, "faiss_rank")
        try:
            faiss_rank = int(faiss_rank_raw)
        except Exception as e:
            raise ValueError(f"Invalid faiss_rank for group_id={group_id}, mention_id={mention_id}: {faiss_rank_raw}") from e

        if group_id not in groups_map:
            groups_map[group_id] = {
                "mention_id": mention_id,
                "context_text": context_text,
                "candidates": [],
            }
        else:
            # consistency checks
            if groups_map[group_id]["mention_id"] != mention_id:
                raise ValueError(
                    f"Inconsistent mention_id within group {group_id}: "
                    f"{groups_map[group_id]['mention_id']} vs {mention_id}"
                )
            if groups_map[group_id]["context_text"] != context_text:
                raise ValueError(
                    f"Inconsistent context_text within group {group_id} for mention_id={mention_id}"
                )

        groups_map[group_id]["candidates"].append(
            Candidate(
                cand_text=cand_text,
                cand_def=cand_def,
                cand_cui=(str(cand_cui) if cand_cui is not None else None),
                is_gold=is_gold,
                is_nil=is_nil,
                faiss_rank=faiss_rank,
            )
        )

    if not groups_map:
        raise RuntimeError("Loaded zero groups from input; aborting")

    groups: List[Group] = []
    for gid, payload in groups_map.items():
        cands: List[Candidate] = payload["candidates"]
        if not cands:
            raise ValueError(f"Group {gid} has zero candidates")
        # Sort by faiss_rank
        cands.sort(key=lambda c: c.faiss_rank)
        gold_indices = [i for i, c in enumerate(cands) if c.is_gold]
        if len(gold_indices) != 1:
            raise ValueError(f"Group {gid} must have exactly one gold candidate; found {len(gold_indices)}")
        groups.append(
            Group(
                group_id=gid,
                mention_id=payload["mention_id"],
                context_text=payload["context_text"],
                candidates=cands,
                gold_index=gold_indices[0],
            )
        )

    return groups


class GroupDataset(Dataset):
    """Dataset over groups; tokenization happens in the collate function."""

    def __init__(self, groups: List[Group]):
        if not groups:
            raise ValueError("GroupDataset requires at least one group")
        self.groups = groups

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int) -> Group:
        return self.groups[idx]


@dataclass
class Batch:
    input_ids: torch.Tensor           # [N, L]
    attention_mask: torch.Tensor      # [N, L]
    token_type_ids: torch.Tensor      # [N, L]
    group_lengths: List[int]          # len == B
    target_indices: torch.Tensor      # [B]
    gold_is_nil: torch.Tensor         # [B] bool
    is_nil_mask: torch.Tensor         # [N] bool (per-candidate)


def build_pair_ids(
    tokenizer: PreTrainedTokenizerFast,
    context_text: str,
    cand_text: str,
    cand_def: str,
    max_length: int,
) -> Tuple[List[int], List[int]]:
    """Construct input_ids and token_type_ids for one (context, candidate) pair.

    Layout: [CLS] ctx [SEP] cand_text [SEP] cand_def [SEP]
    Token type ids: 0 for [CLS] ctx [SEP]; 1 afterwards.
    Truncation strategy: trim cand_def first, then cand_text, then context.
    """
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id
    if cls_id is None or sep_id is None or pad_id is None:
        raise RuntimeError("Tokenizer must define cls_token_id, sep_token_id, and pad_token_id")

    def enc(text: str) -> List[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    ids_ctx = enc(context_text)
    ids_lbl = enc(cand_text)
    ids_def = enc(cand_def if cand_def else "[NO_DEF]")

    # Compose and truncate to max_length
    def total_len(a: List[int], b: List[int], c: List[int]) -> int:
        return 1 + len(a) + 1 + len(b) + 1 + len(c) + 1  # CLS + A + SEP + B + SEP + C + SEP

    while total_len(ids_ctx, ids_lbl, ids_def) > max_length:
        if ids_def:
            ids_def.pop()
            continue
        if ids_lbl:
            ids_lbl.pop()
            continue
        if ids_ctx:
            ids_ctx.pop()
            continue
        # Should not happen; but enforce hard stop
        break

    input_ids: List[int] = [cls_id] + ids_ctx + [sep_id] + ids_lbl + [sep_id] + ids_def + [sep_id]
    # token_type_ids: 0 until first SEP inclusive; 1 afterwards
    type_ids: List[int] = [0] * (1 + len(ids_ctx) + 1) + [1] * (len(ids_lbl) + 1 + len(ids_def) + 1)

    return input_ids, type_ids


def collate_groups(
    batch_groups: List[Group],
    tokenizer: PreTrainedTokenizerFast,
    max_length: int,
) -> Batch:
    """Tokenize and pad a list of groups into a flat candidate batch."""
    input_ids_flat: List[List[int]] = []
    type_ids_flat: List[List[int]] = []
    attn_flat: List[List[int]] = []
    is_nil_flat: List[bool] = []
    group_lengths: List[int] = []
    target_indices: List[int] = []
    gold_is_nil_flags: List[bool] = []

    for grp in batch_groups:
        group_len = len(grp.candidates)
        if group_len == 0:
            raise ValueError(f"Encountered empty group: {grp.group_id}")
        group_lengths.append(group_len)
        target_indices.append(int(grp.gold_index))
        # Record whether the gold candidate for this group is NIL
        gold_cand = grp.candidates[int(grp.gold_index)]
        gold_is_nil_flags.append(bool(gold_cand.is_nil))
        for cand in grp.candidates:
            pair_ids, type_ids = build_pair_ids(
                tokenizer=tokenizer,
                context_text=grp.context_text,
                cand_text=cand.cand_text,
                cand_def=cand.cand_def,
                max_length=max_length,
            )
            # Pad to max_length
            if len(pair_ids) > max_length or len(type_ids) > max_length:
                raise RuntimeError("Internal error: sequence longer than max_length after truncation")
            pad_len = max_length - len(pair_ids)
            input_ids_flat.append(pair_ids + [tokenizer.pad_token_id] * pad_len)
            type_ids_flat.append(type_ids + [0] * pad_len)
            attn_flat.append([1] * len(pair_ids) + [0] * pad_len)
            is_nil_flat.append(bool(cand.is_nil))

    input_ids = torch.tensor(input_ids_flat, dtype=torch.long)
    token_type_ids = torch.tensor(type_ids_flat, dtype=torch.long)
    attention_mask = torch.tensor(attn_flat, dtype=torch.long)
    targets = torch.tensor(target_indices, dtype=torch.long)
    gold_is_nil = torch.tensor(gold_is_nil_flags, dtype=torch.bool)
    is_nil_mask = torch.tensor(is_nil_flat, dtype=torch.bool)
    return Batch(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                 group_lengths=group_lengths, target_indices=targets, gold_is_nil=gold_is_nil,
                 is_nil_mask=is_nil_mask)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_groups(groups: List[Group], train_frac: float, seed: int) -> Tuple[List[Group], List[Group]]:
    if not (0.0 < train_frac < 1.0):
        raise ValueError(f"train_frac must be in (0,1), got {train_frac}")
    rng = random.Random(seed)
    order = list(range(len(groups)))
    rng.shuffle(order)
    cut = int(len(groups) * train_frac)
    idx_train = order[:cut]
    idx_eval = order[cut:]
    train = [groups[i] for i in idx_train]
    eval_ = [groups[i] for i in idx_eval]
    if not train or not eval_:
        raise RuntimeError("Train/eval split produced empty partition; adjust --train_frac")
    return train, eval_


def grouped_loss_and_accuracy(logits: torch.Tensor, group_lengths: List[int], targets: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """Compute sum CE loss across groups and top-1 accuracy.

    logits: [N, 1] scores per candidate
    group_lengths: list of group sizes summing to N
    targets: [B] gold index in each group
    """
    if logits.ndim != 2 or logits.size(1) != 1:
        raise ValueError(f"Expected logits shape [N,1], got {tuple(logits.shape)}")
    offset = 0
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss = torch.zeros((), device=logits.device)
    correct = 0
    B = len(group_lengths)
    for b, glen in enumerate(group_lengths):
        if glen <= 0:
            raise ValueError("Group length must be positive")
        sl = logits[offset:offset + glen, 0].unsqueeze(0)  # [1, glen]
        tgt = targets[b:b+1]  # [1]
        total_loss = total_loss + loss_fn(sl, tgt)
        pred_idx = int(torch.argmax(sl, dim=1).item())
        if pred_idx == int(tgt.item()):
            correct += 1
        offset += glen
    avg_loss = total_loss / float(B)
    acc = correct / float(B)
    return avg_loss, acc


def compute_hinge_loss_and_gap_stats(
    logits: torch.Tensor,
    group_lengths: List[int],
    targets: torch.Tensor,
    gold_is_nil: torch.Tensor,
    margin: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute average hinge loss across groups and gap statistics.

    Hinge per group: max(0, margin - s_gold + s_hardest_negative),
    where s_hardest_negative is the highest non-gold score. Skip hinge for
    groups whose gold is NIL, and for degenerate groups with <2 candidates.

    Returns (avg_hinge_loss_tensor, stats_dict). The avg_hinge_loss is averaged
    over total number of groups (len(group_lengths)), with skipped groups
    contributing zero per the requirement to average over groups.
    """
    if logits.ndim != 2 or logits.size(1) != 1:
        raise ValueError(f"Expected logits shape [N,1], got {tuple(logits.shape)}")
    if len(group_lengths) != int(targets.numel()):
        raise ValueError("group_lengths and targets must have the same number of groups")
    if len(group_lengths) != int(gold_is_nil.numel()):
        raise ValueError("group_lengths and gold_is_nil must have the same number of groups")
    if margin < 0.0:
        raise ValueError(f"hinge margin must be non-negative, got {margin}")

    device = logits.device
    offset = 0
    total_groups = len(group_lengths)
    total_hinge = torch.zeros((), device=device)

    # Stats
    considered = 0
    sum_gap = 0.0
    min_gap: Optional[float] = None
    max_gap: Optional[float] = None
    violations = 0

    for b, glen in enumerate(group_lengths):
        if glen <= 0:
            raise ValueError("Group length must be positive")
        # Skip hinge if no negative candidate or gold is NIL
        skip = False
        if glen < 2:
            skip = True
        if bool(gold_is_nil[b].item()):
            skip = True

        scores = logits[offset:offset + glen, 0]
        tgt = int(targets[b].item())
        if tgt < 0 or tgt >= glen:
            raise ValueError(f"Target index {tgt} out of range for group of length {glen}")
        gold_score = scores[tgt]

        if not skip:
            # Hardest negative
            neg_mask = torch.ones(glen, dtype=torch.bool, device=device)
            neg_mask[tgt] = False
            neg_scores = scores[neg_mask]
            if neg_scores.numel() == 0:
                # No negatives present; skip hinge for this group
                hinge = torch.zeros((), device=device)
                gap_val = float("nan")
            else:
                hardest_neg = torch.max(neg_scores)
                # Gap = s_gold - s_hardest_neg
                gap_val = float((gold_score - hardest_neg).detach().cpu().item())
                raw = float(margin)
                hinge_val = torch.clamp(raw - gold_score + hardest_neg, min=0.0)
                hinge = hinge_val

            # Update stats if we had negatives
            if neg_scores.numel() > 0:
                considered += 1
                if not (gap_val != gap_val):  # filter NaN
                    sum_gap += gap_val
                    min_gap = gap_val if (min_gap is None or gap_val < min_gap) else min_gap
                    max_gap = gap_val if (max_gap is None or gap_val > max_gap) else max_gap
                    if gap_val < float(margin):
                        violations += 1
        else:
            hinge = torch.zeros((), device=device)

        total_hinge = total_hinge + hinge
        offset += glen

    avg_hinge = total_hinge / float(total_groups)
    stats: Dict[str, float] = {
        "hinge_loss": float(avg_hinge.detach().cpu().item()),
        "gap_min": (float(min_gap) if min_gap is not None else None),
        "gap_mean": (float(sum_gap / considered) if considered > 0 else None),
        "gap_max": (float(max_gap) if max_gap is not None else None),
        "pct_violations": (float(violations / considered) if considered > 0 else None),
        "groups_considered": int(considered),
        "total_groups": int(total_groups),
    }
    return avg_hinge, stats


def apply_nil_bias_to_logits(
    logits: torch.Tensor,
    is_nil_mask: torch.Tensor,
    nil_bias: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    """Adjust logits for NIL rows according to mode using a learned bias.

    modes:
      - 'bias':           s_nil = b_nil
      - 'bias_plus_text': s_nil = b_nil + s_text_nil
    """
    if logits.ndim != 2 or logits.size(1) != 1:
        raise ValueError(f"Expected logits shape [N,1], got {tuple(logits.shape)}")
    if is_nil_mask.ndim != 1 or is_nil_mask.numel() != logits.size(0):
        raise ValueError("is_nil_mask must be shape [N] and match logits batch size")
    if mode not in ("bias", "bias_plus_text"):
        raise ValueError(f"Invalid --nil_mode: {mode}")
    mask = is_nil_mask.to(dtype=logits.dtype, device=logits.device).unsqueeze(1)  # [N,1]
    if mode == "bias":
        # Replace NIL rows with bias only
        return logits * (1.0 - mask) + nil_bias.to(logits.device) * mask
    else:
        # Add bias to NIL rows, leave others unchanged
        return logits + nil_bias.to(logits.device) * mask

def compute_faiss_accuracy_at_k(groups: List[Group], ks: Sequence[int]) -> Dict[int, float]:
    """Compute baseline FAISS@K from combined groups using candidate faiss_rank.

    For each group, we locate the gold candidate and read its faiss_rank; a hit@k
    is counted if faiss_rank <= k. If a group does not have exactly one gold
    candidate, raise an error (the input is expected to be well-formed).
    """
    if not groups:
        raise ValueError("No groups provided to compute FAISS baseline accuracy")
    ks_sorted = sorted(int(k) for k in ks)
    counters: Dict[int, int] = {int(k): 0 for k in ks_sorted}
    total = len(groups)
    for grp in groups:
        gold_positions = [i for i, c in enumerate(grp.candidates) if c.is_gold]
        if len(gold_positions) != 1:
            raise ValueError(f"Group {grp.group_id} must have exactly one gold candidate; found {len(gold_positions)}")
        gi = gold_positions[0]
        gold_rank = int(grp.candidates[gi].faiss_rank)
        for k in ks_sorted:
            if gold_rank <= k:
                counters[k] += 1
    return {k: (counters[k] / float(total)) for k in ks_sorted}


def evaluate_loader_topk(
    model: AutoModelForSequenceClassification,
    loader: DataLoader,
    device: torch.device,
    ks: Sequence[int] = (1, 5, 10),
    compute_loss: bool = True,
    hinge_margin: Optional[float] = None,
    nil_mode: str = "bias",
    nil_bias: Optional[torch.Tensor] = None,
) -> Tuple[Optional[float], Dict[int, float], Dict[str, float], Optional[float], Dict[str, Optional[float]]]:
    """Evaluate model on a loader, returning (avg_loss or None, acc@K dict).

    Acc@K is computed exactly by checking whether the gold index is within the
    top-K logits within each group (K is capped by the actual group size).
    """
    ks_sorted = sorted(int(k) for k in ks)
    correct_by_k: Dict[int, int] = {int(k): 0 for k in ks_sorted}
    total_groups = 0
    loss_total = 0.0
    # Hinge accumulators
    hinge_sum = 0.0
    sum_gap = 0.0
    gap_min: Optional[float] = None
    gap_max: Optional[float] = None
    viol = 0
    gh_considered = 0
    gh_total = 0

    nil_total = 0
    nil_correct = 0
    # MRR and flip-rate accumulators
    mrr_sum = 0.0
    flip_eligible = 0
    flip_rescued = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch.input_ids.to(device)
            attn = batch.attention_mask.to(device)
            ttype = batch.token_type_ids.to(device)
            targets = batch.target_indices.to(device)
            outputs = model(input_ids=input_ids, attention_mask=attn, token_type_ids=ttype, return_dict=True)
            logits = outputs.logits  # [N, 1]
            if nil_bias is None:
                raise RuntimeError("Internal error: nil_bias must be provided for evaluation")
            logits = apply_nil_bias_to_logits(logits, batch.is_nil_mask, nil_bias, nil_mode)
            if compute_loss:
                loss, _ = grouped_loss_and_accuracy(logits, batch.group_lengths, targets)
                loss_total += float(loss.item()) * len(batch.group_lengths)
            # Hinge metrics (always compute for logging if margin provided)
            if hinge_margin is not None:
                gold_is_nil = batch.gold_is_nil.to(device)
                hinge_val, stats = compute_hinge_loss_and_gap_stats(
                    logits, batch.group_lengths, targets, gold_is_nil, float(hinge_margin)
                )
                hinge_sum += float(hinge_val.detach().cpu().item()) * len(batch.group_lengths)
                # Aggregate gap stats across batches
                gh_total += len(batch.group_lengths)
                c = int(stats.get("groups_considered", 0) or 0)
                gh_considered += c
                sum_gap += float(stats.get("gap_mean", 0.0) or 0.0) * c
                pct = stats.get("pct_violations")
                if pct is not None:
                    viol += int(round(pct * c))
                gmin = stats.get("gap_min")
                gmax = stats.get("gap_max")
                if gmin is not None:
                    gap_min = gmin if (gap_min is None or gmin < gap_min) else gap_min
                if gmax is not None:
                    gap_max = gmax if (gap_max is None or gmax > gap_max) else gap_max
            # Compute top-K correctness per group
            offset = 0
            for i, glen in enumerate(batch.group_lengths):
                if glen <= 0:
                    raise ValueError("Group length must be positive")
                sl = logits[offset:offset + glen, 0]  # [glen]
                tgt = int(batch.target_indices[i].item())
                if tgt < 0 or tgt >= glen:
                    raise ValueError(f"Target index {tgt} out of range for group of length {glen}")
                # Determine correctness at each requested K (cap by glen)
                for k in ks_sorted:
                    top_k = int(k) if int(k) <= glen else glen
                    # Short-circuit: if top_k == 1 use argmax; else use topk
                    if top_k == 1:
                        pred = int(torch.argmax(sl).item())
                        hit = (pred == tgt)
                    else:
                        top_vals, top_idx = torch.topk(sl, k=top_k, largest=True, sorted=False)
                        hit = bool((top_idx == tgt).any().item())
                    if hit:
                        correct_by_k[int(k)] += 1
                # NIL accuracy (gold NIL, top1 is NIL)
                if bool(batch.gold_is_nil[i].item()):
                    nil_total += 1
                    pred_top1 = int(torch.argmax(sl).item())
                    if pred_top1 == tgt:
                        nil_correct += 1
                # MRR (gold rank among logits, 1-based)
                # rank = 1 + number of candidates with score strictly greater than gold
                gold_score = sl[tgt]
                rank_gold = 1 + int((sl > gold_score).sum().item())
                mrr_sum += 1.0 / float(rank_gold)
                # Flip-rate eligibility: baseline rank 2-5 (i.e., tgt in [1..4]) and non-NIL gold
                if not bool(batch.gold_is_nil[i].item()) and 1 <= tgt <= 4:
                    flip_eligible += 1
                    if int(torch.argmax(sl).item()) == tgt:
                        flip_rescued += 1
                total_groups += 1
                offset += glen
    avg_loss = (loss_total / float(total_groups)) if compute_loss else None
    acc = {k: (correct_by_k[k] / float(total_groups)) for k in ks_sorted}
    # Finalize hinge stats
    hinge_info: Dict[str, float] = {}
    if hinge_margin is not None:
        avg_hinge = hinge_sum / float(total_groups) if total_groups > 0 else 0.0
        gap_mean = (sum_gap / gh_considered) if gh_considered > 0 else None
        pct_violations = (viol / gh_considered) if gh_considered > 0 else None
        hinge_info = {
            "hinge_loss": float(avg_hinge),
            "gap_min": (float(gap_min) if gap_min is not None else None),
            "gap_mean": (float(gap_mean) if gap_mean is not None else None),
            "gap_max": (float(gap_max) if gap_max is not None else None),
            "pct_violations": (float(pct_violations) if pct_violations is not None else None),
            "groups_considered": int(gh_considered),
            "total_groups": int(total_groups),
            "margin": float(hinge_margin),
        }
    nil_acc = (nil_correct / float(nil_total)) if nil_total > 0 else None
    extra_metrics: Dict[str, Optional[float]] = {
        "mrr": (float(mrr_sum / float(total_groups)) if total_groups > 0 else None),
        "flip_rate": (float(flip_rescued / float(flip_eligible)) if flip_eligible > 0 else None),
        "flip_eligible": float(flip_eligible),
    }
    return avg_loss, acc, hinge_info, (float(nil_acc) if nil_acc is not None else None), extra_metrics


def train_loop(
    model: AutoModelForSequenceClassification,
    tokenizer: PreTrainedTokenizerFast,
    train_ds: GroupDataset,
    eval_ds: GroupDataset,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)


    # optimizer = AdamW(model.parameters(), lr=float(args.lr))
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=float(args.lr))


    # Create learned NIL bias as a scalar parameter stored on the model for checkpoints
    if not hasattr(model, "nil_bias"):
        model.nil_bias = nn.Parameter(torch.zeros((), device=device))
        # Ensure optimizer sees the new parameter
        optimizer.add_param_group({"params": [model.nil_bias]})
    if not isinstance(model.nil_bias, torch.Tensor) or model.nil_bias.ndim != 0:
        raise RuntimeError("Model attribute 'nil_bias' must be a scalar tensor parameter")

    def make_loader(ds: GroupDataset) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=int(args.groups_per_batch),
            shuffle=True,
            num_workers=int(args.num_workers),
            collate_fn=lambda batch: collate_groups(batch, tokenizer, int(args.max_length)),
            drop_last=False,
            pin_memory=(device.type == "cuda"),
        )

    train_loader = make_loader(train_ds)
    # Non-shuffling loaders for evaluation metrics on full splits
    train_eval_loader = DataLoader(
        train_ds,
        batch_size=int(args.groups_per_batch),
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=lambda batch: collate_groups(batch, tokenizer, int(args.max_length)),
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=int(args.groups_per_batch),
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=lambda batch: collate_groups(batch, tokenizer, int(args.max_length)),
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )

    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp) and device.type == "cuda")

    best_eval_acc = -1.0
    best_path = None

    for epoch in range(int(args.epochs)):
        model.train()
        running_loss = 0.0
        running_hinge = 0.0
        seen_groups = 0
        for batch in train_loader:
            input_ids = batch.input_ids.to(device)
            attn = batch.attention_mask.to(device)
            ttype = batch.token_type_ids.to(device)
            targets = batch.target_indices.to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                outputs = model(input_ids=input_ids, attention_mask=attn, token_type_ids=ttype, return_dict=True)
                logits = outputs.logits  # [N, num_labels]
                if logits.size(1) != 1:
                    raise RuntimeError(f"Model returned logits with num_labels={logits.size(1)}; expected 1")
                # Adjust NIL rows according to selected mode
                logits = apply_nil_bias_to_logits(logits, batch.is_nil_mask, model.nil_bias, str(getattr(args, "nil_mode", "bias")))
                ce_loss, _ = grouped_loss_and_accuracy(logits, batch.group_lengths, targets)
                # Optional hinge loss term (averaged over groups; skipped for NIL-gold groups)
                if float(args.hinge_weight) > 0.0:
                    hinge_tensor, _ = compute_hinge_loss_and_gap_stats(
                        logits, batch.group_lengths, targets, batch.gold_is_nil.to(device), float(args.hinge_margin)
                    )
                    loss = ce_loss + float(args.hinge_weight) * hinge_tensor
                else:
                    hinge_tensor = torch.zeros((), device=logits.device)
                    loss = ce_loss

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            running_loss += float(ce_loss.item()) * len(batch.group_lengths)
            running_hinge += float(hinge_tensor.detach().cpu().item()) * len(batch.group_lengths)
            seen_groups += len(batch.group_lengths)

        avg_train_loss = running_loss / float(seen_groups)
        avg_train_hinge = running_hinge / float(seen_groups)

        # Eval: compute loss and acc@K on eval set, and acc@K on train set
        model.eval()
        avg_eval_loss, eval_accs, eval_hinge, eval_nil_acc, eval_extra = evaluate_loader_topk(
            model, eval_loader, device, ks=(1, 5, 10), compute_loss=True, hinge_margin=float(args.hinge_margin),
            nil_mode=str(getattr(args, "nil_mode", "bias")), nil_bias=model.nil_bias
        )
        _, train_accs, train_hinge, train_nil_acc, train_extra = evaluate_loader_topk(
            model, train_eval_loader, device, ks=(1, 5, 10), compute_loss=False, hinge_margin=float(args.hinge_margin),
            nil_mode=str(getattr(args, "nil_mode", "bias")), nil_bias=model.nil_bias
        )

        print(json.dumps({
            "epoch": int(epoch+1),
            "avg_train_loss": float(avg_train_loss),
            "avg_eval_loss": float(avg_eval_loss),
            "train_hinge_loss": float(avg_train_hinge),
            "eval_hinge_loss": float(eval_hinge.get("hinge_loss", 0.0) if eval_hinge else 0.0),
            "train_gap_stats": {
                "gap_min": train_hinge.get("gap_min"),
                "gap_mean": train_hinge.get("gap_mean"),
                "gap_max": train_hinge.get("gap_max"),
                "pct_violations": train_hinge.get("pct_violations"),
                "margin": float(args.hinge_margin),
            },
            "eval_gap_stats": {
                "gap_min": eval_hinge.get("gap_min"),
                "gap_mean": eval_hinge.get("gap_mean"),
                "gap_max": eval_hinge.get("gap_max"),
                "pct_violations": eval_hinge.get("pct_violations"),
                "margin": float(args.hinge_margin),
            },
            "train_nil_acc": (float(train_nil_acc) if train_nil_acc is not None else None),
            "eval_nil_acc": (float(eval_nil_acc) if eval_nil_acc is not None else None),
            "nil_bias": float(model.nil_bias.detach().cpu().item()),
            "train_mrr": (float(train_extra.get("mrr")) if train_extra else None),
            "eval_mrr": (float(eval_extra.get("mrr")) if eval_extra else None),
            "train_flip_rate": (float(train_extra.get("flip_rate")) if train_extra else None),
            "eval_flip_rate": (float(eval_extra.get("flip_rate")) if eval_extra else None),
            "train_acc@1": float(train_accs.get(1, 0.0)),
            "train_acc@5": float(train_accs.get(5, 0.0)),
            "train_acc@10": float(train_accs.get(10, 0.0)),
            "eval_acc_top1": float(eval_accs.get(1, 0.0)),
            "eval_acc@1": float(eval_accs.get(1, 0.0)),
            "eval_acc@5": float(eval_accs.get(5, 0.0)),
            "eval_acc@10": float(eval_accs.get(10, 0.0)),
        }))

        # Save checkpoint for this epoch (always). If disk is full (ENOSPC), log and continue.
        try:
            os.makedirs(args.output_dir, exist_ok=True)
            epoch_dir = os.path.join(args.output_dir, f"epoch_{int(epoch+1)}")
            os.makedirs(epoch_dir, exist_ok=True)
            model.save_pretrained(epoch_dir)
            tokenizer.save_pretrained(epoch_dir)
        except OSError as e:
            if isinstance(e, OSError) and getattr(e, 'errno', None) == errno.ENOSPC:
                print(f"[warn] Skipping save for epoch {int(epoch+1)} due to no space left on device: {e}", file=sys.stderr)
            else:
                raise

        # Save best
        if float(eval_accs.get(1, 0.0)) > best_eval_acc:
            best_eval_acc = float(eval_accs.get(1, 0.0))
            os.makedirs(args.output_dir, exist_ok=True)
            save_dir = os.path.join(args.output_dir, "best")
            os.makedirs(save_dir, exist_ok=True)
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            best_path = save_dir

    return {"best_eval_acc": float(best_eval_acc), "best_path": best_path}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Train a cross-encoder on combined CE candidates and contexts")
        # near other add_argument calls
    parser.add_argument("--freeze_n_layers", type=int, default=0,
                        help="Freeze first N transformer blocks of the encoder (0=train all)")
    parser.add_argument("--train_jsonl_gz", type=str, required=False, help="Path to combined training JSONL.GZ (required)")
    parser.add_argument("--eval_jsonl_gz", type=str, required=False, help="Path to combined evaluation JSONL.GZ (required)")
    parser.add_argument("--output_dir", type=str, default=os.path.join("cross_encoder", "ce_runs"))
    parser.add_argument("--encoder_name", type=str, default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--groups_per_batch", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision (CUDA only)")
    # Pairwise hinge augmentation
    parser.add_argument("--hinge_margin", type=float, default=0.2, help="Margin for pairwise hinge loss (gold vs hardest negative)")
    parser.add_argument("--hinge_weight", type=float, default=0.5, help="Weight for hinge loss when added to CE loss")
    # NIL scoring mode
    parser.add_argument("--nil_mode", type=str, default="bias", choices=["bias", "bias_plus_text"], help="How to score NIL rows: bias or bias_plus_text")

    args = parser.parse_args(argv)

    # Fail fast validations
    if not args.train_jsonl_gz or not args.eval_jsonl_gz:
        raise ValueError("You must provide both --train_jsonl_gz and --eval_jsonl_gz. No single-file or fallback mode is supported.")
    if not os.path.exists(args.train_jsonl_gz):
        raise FileNotFoundError(f"Training data file not found: {args.train_jsonl_gz}")
    if not os.path.exists(args.eval_jsonl_gz):
        raise FileNotFoundError(f"Evaluation data file not found: {args.eval_jsonl_gz}")
    if args.max_length <= 8:
        raise ValueError("--max_length must be > 8 to accommodate special tokens")
    if args.hinge_margin < 0.0:
        raise ValueError(f"--hinge_margin must be non-negative, got {args.hinge_margin}")
    if args.hinge_weight < 0.0:
        raise ValueError(f"--hinge_weight must be non-negative, got {args.hinge_weight}")
    if args.nil_mode not in ("bias", "bias_plus_text"):
        raise ValueError(f"--nil_mode must be one of ['bias','bias_plus_text'], got {args.nil_mode}")

    set_seed(int(args.seed))

    # Load groups from explicit train/eval files (no random split)
    train_groups = load_groups(args.train_jsonl_gz)
    eval_groups = load_groups(args.eval_jsonl_gz)

    # Baseline FAISS@K metrics from candidate ranks in the combined files
    baseline_train = compute_faiss_accuracy_at_k(train_groups, ks=(1, 5, 10))
    baseline_eval = compute_faiss_accuracy_at_k(eval_groups, ks=(1, 5, 10))
    print(json.dumps({
        "baseline_faiss_acc_train@1": float(baseline_train.get(1, 0.0)),
        "baseline_faiss_acc_train@5": float(baseline_train.get(5, 0.0)),
        "baseline_faiss_acc_train@10": float(baseline_train.get(10, 0.0)),
        "baseline_faiss_acc_eval@1": float(baseline_eval.get(1, 0.0)),
        "baseline_faiss_acc_eval@5": float(baseline_eval.get(5, 0.0)),
        "baseline_faiss_acc_eval@10": float(baseline_eval.get(10, 0.0)),
    }))

    # Tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.encoder_name)
    # Add special tokens (no auto-fallbacks)
    added = tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})
    config = AutoConfig.from_pretrained(args.encoder_name)
    # Single-score head per candidate
    config.num_labels = 1
    model = AutoModelForSequenceClassification.from_pretrained(args.encoder_name, config=config)

    if added:
        model.resize_token_embeddings(len(tokenizer))
    _freeze_first_n_layers(model, args.freeze_n_layers)
    train_ds = GroupDataset(train_groups)
    eval_ds = GroupDataset(eval_groups)

    metrics = train_loop(model, tokenizer, train_ds, eval_ds, args)

    # Final report
    print(json.dumps({
        "best_eval_acc": float(metrics.get("best_eval_acc", -1.0)),
        "best_model_path": metrics.get("best_path"),
        "output_dir": args.output_dir,
    }))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# 20000mib
# (.venv) bash-5.1$ python -m cross_encoder.train_ce_from_combined   --train_jsonl_gz cross_encoder/combined_train_debug.jsonl.gz   --eval_jsonl_gz  cross_encoder/combined_val_debug.jsonl.gz   --output_dir     cross_encoder/ce_runs   --encoder_name   cambridgeltl/SapBERT-from-PubMedBERT-fulltext   --groups_per_batch 32   --num_worker
# s 16   --amp


# python -m cross_encoder.train_ce_from_combined \
#   --train_jsonl_gz cross_encoder/combined_train_debug.jsonl.gz \
#   --eval_jsonl_gz  cross_encoder/combined_val_debug.jsonl.gz \
#   --output_dir     cross_encoder/ce_runs \
#   --encoder_name   cambridgeltl/SapBERT-from-PubMedBERT-fulltext \
#   --groups_per_batch 32 \
#   --amp \
# |& tee -a cross_encoder/ce_runs/run_log.md