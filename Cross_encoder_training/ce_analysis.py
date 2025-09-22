#!/usr/bin/env python3
"""
Plot Cross-Encoder training metrics from a metrics.jsonl file.

Usage:
  python ce_graphs/test.py --run_dir results/cstm_ce_run

This script expects a file named 'metrics.jsonl' inside --run_dir.
It reads one JSON object per line (one epoch per line), extracts the
cross-encoder metrics produced by cross_encoder/train_ce_from_combined.py,
and saves multiple PNG plots into the same --run_dir.

All plots use epoch on the x-axis. Missing values are skipped.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _load_metrics_lines(metrics_path: Path) -> List[Dict]:
    if not metrics_path.is_file():
        raise FileNotFoundError(f"metrics file not found: {metrics_path}")
    rows: List[Dict] = []
    with metrics_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
            except Exception:
                # Skip malformed lines to be robust
                continue
    # Sort by epoch if available
    try:
        rows.sort(key=lambda r: int(r.get("epoch", 0)))
    except Exception:
        pass
    return rows


def _series(rows: List[Dict], key: str) -> Tuple[List[int], List[float]]:
    xs: List[int] = []
    ys: List[float] = []
    for r in rows:
        if "epoch" not in r:
            continue
        val = r.get(key)
        if val is None:
            continue
        xs.append(int(r["epoch"]))
        try:
            ys.append(float(val))
        except Exception:
            # Skip non-numeric
            xs.pop()
            continue
    return xs, ys


def _nested_series(rows: List[Dict], parent_key: str, child_key: str) -> Tuple[List[int], List[float]]:
    xs: List[int] = []
    ys: List[float] = []
    for r in rows:
        if "epoch" not in r:
            continue
        parent = r.get(parent_key) or {}
        val = None if not isinstance(parent, dict) else parent.get(child_key)
        if val is None:
            continue
        xs.append(int(r["epoch"]))
        try:
            ys.append(float(val))
        except Exception:
            xs.pop()
            continue
    return xs, ys


def _save_lines(
    run_dir: Path,
    title: str,
    y_label: str,
    filename: str,
    series: List[Tuple[str, List[int], List[float]]],
    max_epoch: Optional[int] = None,
) -> None:
    if not series:
        return
    plt.figure(figsize=(8, 5))
    for label, xs, ys in series:
        if max_epoch is not None and xs and ys:
            # Filter points where epoch (x) <= max_epoch
            xs, ys = zip(*[(x, y) for x, y in zip(xs, ys) if int(x) <= int(max_epoch)]) if xs and ys else ([], [])
            xs, ys = list(xs), list(ys)
        if not xs or not ys:
            continue
        plt.plot(xs, ys, marker="o", label=label)
    plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel(y_label)
    plt.grid(True, linestyle=":", linewidth=0.7)
    plt.legend()
    plt.tight_layout()
    out_path = run_dir / filename
    plt.savefig(out_path)
    plt.close()


def make_plots(run_dir: Path, epochs_to_plot: Optional[int] = None) -> None:
    metrics_path = run_dir / "metrics.jsonl"
    # Backward compatibility: allow .json if .jsonl not present
    if not metrics_path.exists():
        alt = run_dir / "metrics.json"
        if alt.exists():
            metrics_path = alt

    rows = _load_metrics_lines(metrics_path)
    if not rows:
        raise RuntimeError(f"No metrics lines loaded from {metrics_path}")

    # Losses
    tr_x, tr_y = _series(rows, "avg_train_loss")
    ev_x, ev_y = _series(rows, "avg_eval_loss")
    _save_lines(
        run_dir,
        title="Cross-Encoder Loss",
        y_label="loss",
        filename="losses.png",
        series=[("Train", tr_x, tr_y), ("Eval", ev_x, ev_y)],
        max_epoch=epochs_to_plot,
    )

    # Hinge loss
    hx, hy = _series(rows, "train_hinge_loss")
    ex, ey = _series(rows, "eval_hinge_loss")
    _save_lines(
        run_dir,
        title="Hinge Loss",
        y_label="hinge",
        filename="hinge_loss.png",
        series=[("Train", hx, hy), ("Eval", ex, ey)],
        max_epoch=epochs_to_plot,
    )

    # Gap stats: train and eval (min/mean/max)
    gmt_x, gmt_min = _nested_series(rows, "train_gap_stats", "gap_min")
    gmt_x2, gmt_mean = _nested_series(rows, "train_gap_stats", "gap_mean")
    gmt_x3, gmt_max = _nested_series(rows, "train_gap_stats", "gap_max")
    _save_lines(
        run_dir,
        title="Gap Stats (Train)",
        y_label="gap (s_gold - s_hardest_neg)",
        filename="gap_stats_train.png",
        series=[("Gap Min", gmt_x, gmt_min), ("Gap Mean", gmt_x2, gmt_mean), ("Gap Max", gmt_x3, gmt_max)],
        max_epoch=epochs_to_plot,
    )

    gme_x, gme_min = _nested_series(rows, "eval_gap_stats", "gap_min")
    gme_x2, gme_mean = _nested_series(rows, "eval_gap_stats", "gap_mean")
    gme_x3, gme_max = _nested_series(rows, "eval_gap_stats", "gap_max")
    _save_lines(
        run_dir,
        title="Gap Stats (Eval)",
        y_label="gap (s_gold - s_hardest_neg)",
        filename="gap_stats_eval.png",
        series=[("Gap Min", gme_x, gme_min), ("Gap Mean", gme_x2, gme_mean), ("Gap Max", gme_x3, gme_max)],
        max_epoch=epochs_to_plot,
    )

    # Gap stats (combined train vs eval): min, mean, max
    _save_lines(
        run_dir,
        title="Gap Min (Train vs Eval)",
        y_label="gap (s_gold - s_hardest_neg)",
        filename="gap_min_train_eval.png",
        series=[("Train", gmt_x, gmt_min), ("Eval", gme_x, gme_min)],
        max_epoch=epochs_to_plot,
    )
    _save_lines(
        run_dir,
        title="Gap Mean (Train vs Eval)",
        y_label="gap (s_gold - s_hardest_neg)",
        filename="gap_mean_train_eval.png",
        series=[("Train", gmt_x2, gmt_mean), ("Eval", gme_x2, gme_mean)],
        max_epoch=epochs_to_plot,
    )
    _save_lines(
        run_dir,
        title="Gap Max (Train vs Eval)",
        y_label="gap (s_gold - s_hardest_neg)",
        filename="gap_max_train_eval.png",
        series=[("Train", gmt_x3, gmt_max), ("Eval", gme_x3, gme_max)],
        max_epoch=epochs_to_plot,
    )

    # Gap violation percent
    pvx_t, pvy_t = _nested_series(rows, "train_gap_stats", "pct_violations")
    pvx_e, pvy_e = _nested_series(rows, "eval_gap_stats", "pct_violations")
    _save_lines(
        run_dir,
        title="Gap Violations (%)",
        y_label="percent",
        filename="gap_pct_violations.png",
        series=[("Train", pvx_t, [v * 100 if v <= 1.0 else v for v in pvy_t]),
                ("Eval", pvx_e, [v * 100 if v <= 1.0 else v for v in pvy_e])],
        max_epoch=epochs_to_plot,
    )

    # NIL accuracy
    nx_t, ny_t = _series(rows, "train_nil_acc")
    nx_e, ny_e = _series(rows, "eval_nil_acc")
    _save_lines(
        run_dir,
        title="NIL Accuracy",
        y_label="accuracy",
        filename="nil_accuracy.png",
        series=[("Train", nx_t, ny_t), ("Eval", nx_e, ny_e)],
        max_epoch=epochs_to_plot,
    )

    # MRR
    mx_t, my_t = _series(rows, "train_mrr")
    mx_e, my_e = _series(rows, "eval_mrr")
    _save_lines(
        run_dir,
        title="MRR",
        y_label="mrr",
        filename="mrr.png",
        series=[("Train", mx_t, my_t), ("Eval", mx_e, my_e)],
        max_epoch=epochs_to_plot,
    )

    # Flip-rate
    fx_t, fy_t = _series(rows, "train_flip_rate")
    fx_e, fy_e = _series(rows, "eval_flip_rate")
    _save_lines(
        run_dir,
        title="Flip Rate",
        y_label="flip_rate",
        filename="flip_rate.png",
        series=[("Train", fx_t, fy_t), ("Eval", fx_e, fy_e)],
        max_epoch=epochs_to_plot,
    )

    # Accuracy@K (train/eval)
    a1x_t, a1y_t = _series(rows, "train_acc@1")
    a5x_t, a5y_t = _series(rows, "train_acc@5")
    a10x_t, a10y_t = _series(rows, "train_acc@10")
    a1x_e, a1y_e = _series(rows, "eval_acc@1")
    a5x_e, a5y_e = _series(rows, "eval_acc@5")
    a10x_e, a10y_e = _series(rows, "eval_acc@10")
    _save_lines(
        run_dir,
        title="Accuracy@K",
        y_label="accuracy",
        filename="accuracy_at_k.png",
        series=[
            ("Train@1", a1x_t, a1y_t), ("Train@5", a5x_t, a5y_t), ("Train@10", a10x_t, a10y_t),
            ("Eval@1", a1x_e, a1y_e), ("Eval@5", a5x_e, a5y_e), ("Eval@10", a10x_e, a10y_e),
        ],
        max_epoch=epochs_to_plot,
    )

    # Accuracy@K (separate): train only and eval only
    _save_lines(
        run_dir,
        title="Accuracy@K (Train)",
        y_label="accuracy",
        filename="accuracy_at_k_train.png",
        series=[
            ("@1", a1x_t, a1y_t), ("@5", a5x_t, a5y_t), ("@10", a10x_t, a10y_t),
        ],
        max_epoch=epochs_to_plot,
    )
    _save_lines(
        run_dir,
        title="Accuracy@K (Eval)",
        y_label="accuracy",
        filename="accuracy_at_k_eval.png",
        series=[
            ("@1", a1x_e, a1y_e), ("@5", a5x_e, a5y_e), ("@10", a10x_e, a10y_e),
        ],
        max_epoch=epochs_to_plot,
    )

    # NIL bias
    bx, by = _series(rows, "nil_bias")
    _save_lines(
        run_dir,
        title="NIL Bias",
        y_label="nil_bias",
        filename="nil_bias.png",
        series=[("NIL Bias", bx, by)],
        max_epoch=epochs_to_plot,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot Cross-Encoder metrics from metrics.jsonl")
    ap.add_argument("--run_dir", required=True, help="Directory that contains metrics.jsonl; plots will be saved here")
    ap.add_argument("--epochs_to_plot", type=int, default=None, help="Limit X-axis to first N epochs across all plots")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {run_dir}")
    make_plots(run_dir, epochs_to_plot=args.epochs_to_plot)
    print(f"Saved plots to: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



