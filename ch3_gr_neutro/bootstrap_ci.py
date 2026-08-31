"""Bootstrap 95% CI on the headline metrics for a given run/ensemble.

Resamples test indices with replacement B times, computes weighted F1, mean
concept F1, and probe weighted F1 each time, then reports 2.5/97.5 percentiles.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rd = Path(args.run_dir)
    pred = torch.load(rd / "predictions.pt", map_location="cpu", weights_only=False)
    ev = json.loads((rd / "eval.json").read_text())

    cls_test_p = pred["test"]["class_logits"].sigmoid().numpy()
    con_test_p = pred["test"]["concept_logits"].sigmoid().numpy()
    y_test = pred["test"]["labels"].numpy().astype(int)
    ct_test = (pred["test"]["concept_targets"].numpy() >= 0.5).astype(int)

    cls_thr = np.array(ev["classification_tuned"]["thresholds"])
    con_thr = np.array(ev["concept_tuned"]["thresholds"])

    pred_cls = (cls_test_p >= cls_thr[None, :]).astype(int)
    pred_con = (con_test_p >= con_thr[None, :]).astype(int)

    n = len(y_test)
    rng = np.random.default_rng(args.seed)
    rec = dict(weighted_f1=[], macro_f1=[], mean_concept_f1=[])
    for b in range(args.n_boot):
        idx = rng.integers(0, n, size=n)
        rec["weighted_f1"].append(
            f1_score(y_test[idx], pred_cls[idx], average="weighted", zero_division=0))
        rec["macro_f1"].append(
            f1_score(y_test[idx], pred_cls[idx], average="macro", zero_division=0))
        per_concept = [f1_score(ct_test[idx, k], pred_con[idx, k], zero_division=0)
                        for k in range(con_test_p.shape[1])]
        rec["mean_concept_f1"].append(float(np.mean(per_concept)))

    print(f"[boot] n_test={n}  n_boot={args.n_boot}")
    for k, v in rec.items():
        v = np.array(v)
        print(f"  {k:20s}  point={v.mean():.4f}  median={np.median(v):.4f}  "
              f"95% CI=[{np.percentile(v, 2.5):.4f}, {np.percentile(v, 97.5):.4f}]")


if __name__ == "__main__":
    main()
