"""Optimize per-class thresholds to MAXIMIZE THE MINIMUM class F1.

The default threshold tuning maximizes each class F1 independently. This script
instead finds a threshold vector that maximizes min_k F1_k, which is the metric
the user actually cares about ("≥0.90 on all classes").

Usage: python min_class_optimize.py --run_dir <path>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score


def evaluate_thr(probs, y, thr):
    pred = (probs >= thr[None, :]).astype(int)
    return [f1_score(y[:, k], pred[:, k], zero_division=0) for k in range(y.shape[1])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--side", choices=["class", "concept"], default="class")
    args = ap.parse_args()
    rd = Path(args.run_dir)
    pred = torch.load(rd / "predictions.pt", map_location="cpu", weights_only=False)
    if args.side == "class":
        val_p = pred["val"]["class_logits"].sigmoid().numpy()
        test_p = pred["test"]["class_logits"].sigmoid().numpy()
        y_val = pred["val"]["labels"].numpy().astype(int)
        y_test = pred["test"]["labels"].numpy().astype(int)
        names = pred["class_names"]
    else:
        val_p = pred["val"]["concept_logits"].sigmoid().numpy()
        test_p = pred["test"]["concept_logits"].sigmoid().numpy()
        y_val = (pred["val"]["concept_targets"].numpy() >= 0.5).astype(int)
        y_test = (pred["test"]["concept_targets"].numpy() >= 0.5).astype(int)
        names = pred["concepts"]

    K = y_val.shape[1]
    grid = np.linspace(0.05, 0.95, 91)

    # 1) Default: per-class independent F1-maximization on val
    indep = np.full(K, 0.5)
    for k in range(K):
        if y_val[:, k].sum() == 0: continue
        f1s = [f1_score(y_val[:, k], (val_p[:, k] >= t).astype(int), zero_division=0) for t in grid]
        indep[k] = grid[int(np.argmax(f1s))]
    indep_test = evaluate_thr(test_p, y_test, indep)

    # 2) Maximin: optimize for min F1 across classes via 1-D coordinate ascent
    # Start from indep, then for each class try all threshold values and pick the
    # one that maximizes min F1 across classes (ties broken by sum F1).
    cur = indep.copy()
    for _ in range(3):  # 3 sweeps
        for k in range(K):
            if y_val[:, k].sum() == 0: continue
            best_t = cur[k]
            best_min = -1; best_sum = -1
            for t in grid:
                cur[k] = t
                vf1s = evaluate_thr(val_p, y_val, cur)
                m = min(vf1s); s = sum(vf1s)
                if m > best_min or (m == best_min and s > best_sum):
                    best_min = m; best_sum = s; best_t = t
            cur[k] = best_t
    maximin = cur.copy()
    maximin_test = evaluate_thr(test_p, y_test, maximin)

    print("\n=== Independent per-class F1 max ===")
    print("thr:", indep.round(3).tolist())
    print("test F1:", [f"{n}={f:.3f}" for n, f in zip(names, indep_test)])
    print(f"  min={min(indep_test):.3f}  mean={np.mean(indep_test):.3f}  weighted={f1_score(y_test, (test_p >= indep[None,:]).astype(int), average='weighted', zero_division=0):.3f}")

    print("\n=== Maximin (max-min F1) per-class threshold ===")
    print("thr:", maximin.round(3).tolist())
    print("test F1:", [f"{n}={f:.3f}" for n, f in zip(names, maximin_test)])
    print(f"  min={min(maximin_test):.3f}  mean={np.mean(maximin_test):.3f}  weighted={f1_score(y_test, (test_p >= maximin[None,:]).astype(int), average='weighted', zero_division=0):.3f}")


if __name__ == "__main__":
    main()
