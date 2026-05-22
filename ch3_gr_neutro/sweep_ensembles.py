"""Try several ensemble combinations and report tuned classification + concept F1
plus probe accuracy. CPU-only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

OUT = Path("/gpfs/workdir/mouaddenn/thesis/ch3_gr_neutro/outputs")


def load(t):
    return torch.load(OUT / t / "predictions.pt", map_location="cpu", weights_only=False)


def ensemble(tags, weights=None):
    blobs = [load(t) for t in tags]
    if weights is None:
        weights = [1.0] * len(blobs)
    weights = np.array(weights) / np.sum(weights)

    def to_list(x):
        return x.tolist() if hasattr(x, "tolist") else list(x)

    ref = to_list(blobs[0]["test_idx"])
    for b in blobs[1:]:
        assert to_list(b["test_idx"]) == ref, "splits differ"

    cls_test = sum(w * b["test"]["class_logits"].sigmoid().numpy() for w, b in zip(weights, blobs))
    con_test = sum(w * b["test"]["concept_logits"].sigmoid().numpy() for w, b in zip(weights, blobs))
    cls_val = sum(w * b["val"]["class_logits"].sigmoid().numpy() for w, b in zip(weights, blobs))
    con_val = sum(w * b["val"]["concept_logits"].sigmoid().numpy() for w, b in zip(weights, blobs))

    y_test = blobs[0]["test"]["labels"].numpy().astype(int)
    y_val = blobs[0]["val"]["labels"].numpy().astype(int)
    ct_test = (blobs[0]["test"]["concept_targets"].numpy() >= 0.5).astype(int)
    ct_val = (blobs[0]["val"]["concept_targets"].numpy() >= 0.5).astype(int)

    grid = np.linspace(0.05, 0.95, 19)
    # Per-class threshold tuning on val
    cls_thr = np.full(y_val.shape[1], 0.5)
    for k in range(y_val.shape[1]):
        if y_val[:, k].sum() == 0:
            continue
        f1s = [f1_score(y_val[:, k], (cls_val[:, k] >= t).astype(int), zero_division=0) for t in grid]
        cls_thr[k] = grid[int(np.argmax(f1s))]
    pred_cls = (cls_test >= cls_thr[None, :]).astype(int)
    wf1 = f1_score(y_test, pred_cls, average="weighted", zero_division=0)
    mf1 = f1_score(y_test, pred_cls, average="macro", zero_division=0)

    # Per-concept threshold tuning
    con_thr = np.full(con_val.shape[1], 0.5)
    for k in range(con_val.shape[1]):
        if ct_val[:, k].sum() == 0:
            continue
        f1s = [f1_score(ct_val[:, k], (con_val[:, k] >= t).astype(int), zero_division=0) for t in grid]
        con_thr[k] = grid[int(np.argmax(f1s))]
    pred_con = (con_test >= con_thr[None, :]).astype(int)
    cf1 = float(np.mean([f1_score(ct_test[:, k], pred_con[:, k], zero_division=0) for k in range(con_test.shape[1])]))

    # Probe (LR on concepts -> class). Use first blob train if all share split.
    probe_wf1 = float("nan"); probe_sa = float("nan")
    if "train" in blobs[0]:
        Xtr = sum(w * b["train"]["concept_logits"].sigmoid().numpy() for w, b in zip(weights, blobs)) if all("train" in b for b in blobs) else blobs[0]["train"]["concept_logits"].sigmoid().numpy()
        ytr = blobs[0]["train"]["labels"].numpy().astype(int)
        clf = OneVsRestClassifier(LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", n_jobs=-1), n_jobs=-1)
        clf.fit(Xtr, ytr)
        probe_pred = clf.predict(con_test)
        probe_wf1 = float(f1_score(y_test, probe_pred, average="weighted", zero_division=0))
        probe_sa = float((probe_pred == y_test).all(axis=1).mean())
    return wf1, mf1, cf1, probe_wf1, probe_sa


cfgs = [
    ("top5 (current)", ["r1v14_kitchen_s2024", "r1v10_lc01_hard_long_s2024",
                          "r1v13_focal_lc01_long_s2024", "r1v8_unfreeze9_hard_lc2_long_s2024",
                          "r1v11_lc0_hard_long_s2024"]),
    ("top3 low-lam", ["r1v14_kitchen_s2024", "r1v10_lc01_hard_long_s2024",
                       "r1v13_focal_lc01_long_s2024"]),
    ("top4 mixed-lam", ["r1v14_kitchen_s2024", "r1v10_lc01_hard_long_s2024",
                          "r1v13_focal_lc01_long_s2024", "r1v8_unfreeze9_hard_lc2_long_s2024"]),
    ("top2 best", ["r1v14_kitchen_s2024", "r1v10_lc01_hard_long_s2024"]),
    ("top4 + r1v9", ["r1v14_kitchen_s2024", "r1v10_lc01_hard_long_s2024",
                       "r1v13_focal_lc01_long_s2024", "r1v9_focal_ema_hard_lc2_long_s2024"]),
    ("weighted v14*2", ["r1v14_kitchen_s2024", "r1v14_kitchen_s2024",
                          "r1v10_lc01_hard_long_s2024", "r1v13_focal_lc01_long_s2024"]),
    ("top6 broad", ["r1v14_kitchen_s2024", "r1v10_lc01_hard_long_s2024",
                      "r1v13_focal_lc01_long_s2024", "r1v8_unfreeze9_hard_lc2_long_s2024",
                      "r1v9_focal_ema_hard_lc2_long_s2024", "r1v5_hard_lc2_long_s2024"]),
]
print("%-28s  %-10s  %-9s  %-10s  %-9s  %-9s" % ("config", "weighted", "macro", "concept", "probe_wf1", "probe_sa"))
for name, tags in cfgs:
    try:
        w, m, c, p, s = ensemble(tags)
        print("%-28s  %.4f      %.4f     %.4f      %.4f     %.4f" % (name, w, m, c, p, s))
    except Exception as e:
        print("%-28s  err: %s" % (name, e))
