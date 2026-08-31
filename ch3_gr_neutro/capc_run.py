"""
Concept-Anchored Prototype Classification (CAPC).

Inference-time classifier: predicted class = argmin_k d(c_hat(x), mu_k)
where mu_k = R[k] (the textbook class-to-concept matrix row), in [0,1]^M.

Distances: Euclidean, Mahalanobis (Sigma = empirical cov of c_hat on GR-Neutro
TRAIN), cosine.

Multi-label variants:
  (A) Independent per-class threshold: y_k = 1[d_k < tau_k], tau tuned on val
      to maximize W-F1.
  (B) Softmax over class prototypes (single-label argmin) for top-1.

Reports for two architectures (B_kitchen_s42 joint kitchen; cbm_joint_s42 pure CBM):
  * GR-Neutro test: top-1 accuracy, single-label W-F1, multi-label W-F1.
  * MLL-23 segmented neutrophils: Normal-recall (Normal in top-1).
  * MLL-23 OOV cells: top-1 distribution + any-abn fraction (top-1 != Normal).
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path("/gpfs/workdir/mouaddenn/thesis/ch3_gr_neutro")
OUT = ROOT / "outputs" / "capc"
OUT.mkdir(parents=True, exist_ok=True)
CFG = json.loads((ROOT / "concept_config_gr_neutro.json").read_text())

CLASS_NAMES = CFG["gr_neutro_classes"]
CONCEPTS = CFG["concepts"]
K = len(CLASS_NAMES)
M = len(CONCEPTS)
NORMAL_IDX = CFG["normal_class_index"]

R = np.array(CFG["class_to_concept_matrix"]["matrix"], dtype=np.float32)  # (K, M)
assert R.shape == (K, M), R.shape

# ----- MLL-23 mapping for OOV detection (only segmented neutrophil is in-vocab as "Normal") -----
IN_VOCAB_SOURCES = {"neutrophil_segmented"}  # the only direct-Normal match (band is a different protocol)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def euclidean(c, mu):
    # c: (N, M), mu: (K, M) -> (N, K)
    diff = c[:, None, :] - mu[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def mahalanobis(c, mu, sigma_inv):
    # c: (N, M), mu: (K, M), sigma_inv: (M, M) -> (N, K)
    diff = c[:, None, :] - mu[None, :, :]  # (N, K, M)
    tmp = diff @ sigma_inv  # (N, K, M)
    d2 = np.einsum("nkm,nkm->nk", tmp, diff)
    return np.sqrt(np.maximum(d2, 0.0))


def cosine_dist(c, mu):
    cn = c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-12)
    mn = mu / (np.linalg.norm(mu, axis=1, keepdims=True) + 1e-12)
    return 1.0 - cn @ mn.T


def compute_distances(c, R, sigma_inv, metric):
    if metric == "euclidean":
        return euclidean(c, R)
    if metric == "mahalanobis":
        return mahalanobis(c, R, sigma_inv)
    if metric == "cosine":
        return cosine_dist(c, R)
    raise ValueError(metric)


def w_f1_multilabel(d, labels, taus):
    # y_k = 1[d_k < tau_k]
    pred = (d < taus[None, :]).astype(np.int32)
    if pred.sum() == 0 and labels.sum() == 0:
        return 1.0
    return f1_score(labels, pred, average="weighted", zero_division=0)


def tune_taus(d_val, y_val):
    """Per-class threshold tuning to maximize binary F1 per class (weighted F1 = sum)."""
    taus = np.zeros(K, dtype=np.float32)
    # Search over each class independently
    for k in range(K):
        d_k = d_val[:, k]
        y_k = y_val[:, k]
        # Candidate thresholds = midpoints between sorted d_k
        sorted_d = np.unique(d_k)
        if len(sorted_d) > 200:
            qs = np.linspace(0, 100, 201)
            sorted_d = np.unique(np.percentile(d_k, qs))
        best_tau = float(np.median(d_k))
        best_f1 = -1.0
        for tau in sorted_d:
            pred = (d_k < tau).astype(np.int32)
            if pred.sum() == 0 and y_k.sum() == 0:
                f1 = 1.0
            else:
                f1 = f1_score(y_k, pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_tau = float(tau)
        taus[k] = best_tau
    return taus


def load_split(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    out = {}
    for sp in ["train", "val", "test"]:
        c_logits = d[sp]["concept_logits"].numpy()
        c_hat = sigmoid(c_logits)
        labels = d[sp]["labels"].numpy().astype(np.int32)  # (N, K) multi-hot
        out[sp] = {"c_hat": c_hat, "labels": labels, "class_logits": d[sp]["class_logits"].numpy()}
    return out


def single_label_top1(d_dist):
    return d_dist.argmin(axis=1)


def evaluate_architecture(name, pred_path, mll23_npz_path):
    print(f"\n===== {name} =====", flush=True)
    splits = load_split(pred_path)
    c_train = splits["train"]["c_hat"]  # (N_train, M)
    c_val = splits["val"]["c_hat"]
    c_test = splits["test"]["c_hat"]
    y_val = splits["val"]["labels"]
    y_test = splits["test"]["labels"]

    # Sigma on TRAIN
    sigma = np.cov(c_train.T) + 1e-4 * np.eye(M)
    sigma_inv = np.linalg.inv(sigma)

    # Single-label "true" class = argmax of multi-hot labels (Normal-only or first abn);
    # For W-F1 we use the argmax for single-label, multi-hot for multi-label.
    y_test_sl = y_test.argmax(axis=1)
    y_val_sl = y_val.argmax(axis=1)

    results = {"architecture": name, "metrics": {}}

    best_metric_grneutro = None
    best_wf1_ml = -1.0
    metric_summary = {}

    for metric in ["euclidean", "mahalanobis", "cosine"]:
        d_val = compute_distances(c_val, R, sigma_inv, metric)
        d_test = compute_distances(c_test, R, sigma_inv, metric)

        # Top-1
        pred_test_sl = single_label_top1(d_test)
        top1 = float(accuracy_score(y_test_sl, pred_test_sl))
        wf1_sl = float(f1_score(y_test_sl, pred_test_sl, average="weighted", zero_division=0))

        # Multi-label thresholds tuned on val
        taus = tune_taus(d_val, y_val)
        wf1_ml = float(w_f1_multilabel(d_test, y_test, taus))

        metric_summary[metric] = {
            "grneutro_test_top1_acc": top1,
            "grneutro_test_w_f1_single_label": wf1_sl,
            "grneutro_test_w_f1_multilabel_tuned": wf1_ml,
            "tuned_taus": taus.tolist(),
        }
        # Picking criterion: best multi-label W-F1 on TEST? Spec says "best on val".
        # So re-evaluate val W-F1 to select.
        wf1_ml_val = float(w_f1_multilabel(d_val, y_val, taus))
        metric_summary[metric]["grneutro_val_w_f1_multilabel_tuned"] = wf1_ml_val
        if wf1_ml_val > best_wf1_ml:
            best_wf1_ml = wf1_ml_val
            best_metric_grneutro = metric

    # ----- MLL-23 evaluation with the BEST metric -----
    metric = best_metric_grneutro
    print(f"Best distance metric on GR-Neutro val (multi-label W-F1): {metric}", flush=True)
    taus = np.asarray(metric_summary[metric]["tuned_taus"], dtype=np.float32)

    mll23 = np.load(mll23_npz_path, allow_pickle=True)
    c_mll23 = mll23["con_scores"].astype(np.float32)  # already sigmoid scores
    sources = mll23["sources"].astype(str)

    # Segmented neutrophils => "Normal" expected
    seg_mask = sources == "neutrophil_segmented"
    band_mask = sources == "neutrophil_band"
    oov_mask = ~(seg_mask | band_mask)

    d_seg = compute_distances(c_mll23[seg_mask], R, sigma_inv, metric)
    pred_seg = d_seg.argmin(axis=1)
    normal_recall_seg = float((pred_seg == NORMAL_IDX).mean()) if seg_mask.any() else None
    seg_top1_counts = {c: int((pred_seg == i).sum()) for i, c in enumerate(CLASS_NAMES)}
    seg_n = int(seg_mask.sum())

    # OOV cells: classes not in vocab (no Normal-expected, no in-vocab match)
    d_oov = compute_distances(c_mll23[oov_mask], R, sigma_inv, metric)
    pred_oov = d_oov.argmin(axis=1)
    oov_top1_counts = {c: int((pred_oov == i).sum()) for i, c in enumerate(CLASS_NAMES)}
    oov_anyabn_frac = float((pred_oov != NORMAL_IDX).mean()) if oov_mask.any() else None
    oov_n = int(oov_mask.sum())

    # Per-source OOV breakdown
    per_source = {}
    for src in np.unique(sources[oov_mask]):
        idx = sources == src
        d_s = compute_distances(c_mll23[idx], R, sigma_inv, metric)
        p_s = d_s.argmin(axis=1)
        per_source[src] = {
            "n": int(idx.sum()),
            "top1_counts": {c: int((p_s == i).sum()) for i, c in enumerate(CLASS_NAMES)},
            "any_abn_frac": float((p_s != NORMAL_IDX).mean()),
        }

    results["best_distance_metric"] = metric
    results["metric_summary"] = metric_summary
    results["mll23"] = {
        "segmented_neutrophils": {
            "n_cells": seg_n,
            "normal_recall_capc": normal_recall_seg,
            "top1_counts": seg_top1_counts,
        },
        "oov_cells": {
            "n_cells": oov_n,
            "top1_counts": oov_top1_counts,
            "any_abn_fraction": oov_anyabn_frac,
            "per_source": per_source,
        },
    }
    return results


def main():
    archs = [
        (
            "B_kitchen_s42",
            ROOT / "outputs" / "B_kitchen_s42" / "predictions.pt",
            ROOT / "outputs" / "mll23_eval" / "per_cell_predictions.npz",
        ),
        (
            "cbm_joint_s42",
            ROOT / "outputs" / "cbm_joint_s42" / "predictions.pt",
            ROOT / "outputs" / "mll23_cross_model" / "cbm_joint_s42" / "per_cell_predictions.npz",
        ),
    ]
    all_results = {"class_names": CLASS_NAMES, "concepts": CONCEPTS, "architectures": {}}
    for name, pp, mp in archs:
        res = evaluate_architecture(name, pp, mp)
        all_results["architectures"][name] = res

    out_json = OUT / "capc_results.json"
    out_json.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved {out_json}", flush=True)

    # Pretty summary
    print("\n========== SUMMARY ==========", flush=True)
    for name, res in all_results["architectures"].items():
        m = res["best_distance_metric"]
        ms = res["metric_summary"][m]
        print(f"{name}: best metric={m}")
        print(
            f"  GR-Neutro test: top-1 acc={ms['grneutro_test_top1_acc']:.4f}, "
            f"W-F1(single)={ms['grneutro_test_w_f1_single_label']:.4f}, "
            f"W-F1(multi-tuned)={ms['grneutro_test_w_f1_multilabel_tuned']:.4f}"
        )
        seg = res["mll23"]["segmented_neutrophils"]
        oov = res["mll23"]["oov_cells"]
        print(
            f"  MLL-23 segmented neutrophils (n={seg['n_cells']}): Normal-recall (CAPC) = {seg['normal_recall_capc']:.4f}"
        )
        print(
            f"  MLL-23 OOV cells (n={oov['n_cells']}): any-abn fraction = {oov['any_abn_fraction']:.4f}"
        )


if __name__ == "__main__":
    main()
