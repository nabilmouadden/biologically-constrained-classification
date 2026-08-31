"""Post-hoc evaluation: conformal coverage, completeness probe, violation rate.

Reads outputs/<tag>/predictions.pt produced by train.py. Adds three things on
top of the headline F1 / mean-AP that train.py already records:

  1. Conformal prediction (multi-label split conformal): per-class threshold
     calibrated on val so that test coverage of true positives is >= 1-alpha.
  2. Concept completeness probe: linear LR concept_logits -> class label.
     Tests "do the concepts contain enough info to classify?". This is the
     standard 'concept completeness' metric (Yeh et al. 2020).
  3. Empirical co-occurrence vs prior C: how close is the observed test-set
     concept co-occurrence to the prior matrix C?

Outputs eval.json next to predictions.pt.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (f1_score, average_precision_score,
                              balanced_accuracy_score, accuracy_score)
from sklearn.multiclass import OneVsRestClassifier


# ----------------- conformal prediction (multi-label split conformal) --------
def conformal_per_class_thresholds(val_probs: np.ndarray, val_y: np.ndarray,
                                    alpha: float = 0.05) -> np.ndarray:
    """Per-class threshold t_k such that on val, P(p_k >= t_k | y_k = 1) >= 1-alpha.

    A test prediction set keeps class k iff p_k >= t_k. For positives, this
    guarantees marginal coverage >= 1-alpha class by class (modulo finite-sample
    variation). For class-conditional coverage, this is the natural multi-label
    analogue of split conformal sets.
    """
    K = val_probs.shape[1]
    thr = np.full(K, 0.5)
    for k in range(K):
        pos = val_probs[val_y[:, k] == 1, k]
        if len(pos) == 0:
            thr[k] = 0.5
            continue
        # For coverage 1-alpha of positives, threshold = (alpha)-quantile of positive probs.
        # i.e. only fail to cover the lowest-probability alpha fraction of positives.
        n = len(pos)
        # Conservative finite-sample correction: use the floor((alpha)*(n+1))-th order stat.
        idx = max(0, int(np.floor(alpha * (n + 1))) - 1)
        sorted_pos = np.sort(pos)
        thr[k] = float(sorted_pos[idx]) if idx < n else 0.0
    return thr


def conformal_eval(val_probs, val_y, test_probs, test_y, alpha: float = 0.05):
    thr = conformal_per_class_thresholds(val_probs, val_y, alpha=alpha)
    pred_set = (test_probs >= thr).astype(int)         # (N, K)
    # Per-class coverage: for each class, fraction of positives that ended up in the set.
    per_class_cov = []
    for k in range(test_probs.shape[1]):
        pos_mask = test_y[:, k] == 1
        if pos_mask.sum() == 0:
            per_class_cov.append(float("nan"))
        else:
            per_class_cov.append(float(pred_set[pos_mask, k].mean()))
    # Marginal positive coverage = expected over all (sample, class) where y=1.
    pos_idx = test_y == 1
    marginal_cov = float(pred_set[pos_idx].mean()) if pos_idx.any() else float("nan")
    set_size = pred_set.sum(axis=1)                    # (N,)
    return dict(
        alpha=alpha,
        thresholds=thr.tolist(),
        per_class_coverage=per_class_cov,
        marginal_positive_coverage=marginal_cov,
        mean_set_size=float(set_size.mean()),
        median_set_size=float(np.median(set_size)),
        max_set_size=int(set_size.max()),
        min_set_size=int(set_size.min()),
    )


# ----------------- concept completeness probe -------------------------------
def completeness_probe(con_logits_train, y_train, con_logits_test, y_test):
    """Train OneVsRest LR on concept logits -> classify into class labels.

    Multi-label: report macro/micro/weighted F1 + mean-AP. The headline
    'completeness probe accuracy' = exact-match (subset) accuracy on test.
    """
    Xtr = torch.sigmoid(torch.tensor(con_logits_train)).numpy()
    Xte = torch.sigmoid(torch.tensor(con_logits_test)).numpy()
    base = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", n_jobs=-1)
    clf = OneVsRestClassifier(base, n_jobs=-1)
    clf.fit(Xtr, y_train)
    probs = clf.predict_proba(Xte)
    pred = (probs >= 0.5).astype(int)
    metrics = dict(
        macro_f1=float(f1_score(y_test, pred, average="macro", zero_division=0)),
        weighted_f1=float(f1_score(y_test, pred, average="weighted", zero_division=0)),
        subset_accuracy=float((pred == y_test).all(axis=1).mean()),
        mean_per_class_accuracy=float((pred == y_test).mean(axis=0).mean()),
    )
    aps = []
    for k in range(y_test.shape[1]):
        if 0 < y_test[:, k].sum() < len(y_test):
            aps.append(average_precision_score(y_test[:, k], probs[:, k]))
    metrics["mean_ap"] = float(np.nanmean(aps)) if aps else float("nan")
    return metrics


# ----------------- empirical vs prior C cooccurrence -------------------------
def empirical_cooccurrence(concept_probs: np.ndarray, threshold: float = 0.5):
    """For each pair (i,j), empirical p(both active) on test."""
    pred = (concept_probs >= threshold).astype(float)
    return (pred.T @ pred) / max(len(pred), 1)


def cooccurrence_vs_prior(concept_probs: np.ndarray, prior_C: np.ndarray,
                          threshold: float = 0.5):
    cooc = empirical_cooccurrence(concept_probs, threshold)
    K = cooc.shape[0]
    # Frobenius distance between (cooc - mean) and the prior C off-diagonal — informative
    # but not required for the headline; we just report both matrices.
    off_mask = ~np.eye(K, dtype=bool)
    return dict(
        empirical_cooccurrence=cooc.tolist(),
        prior_C=prior_C.tolist(),
        off_diag_pearson_corr=float(np.corrcoef(cooc[off_mask], prior_C[off_mask])[0, 1]),
    )


def violation_rate_soft(con_probs: np.ndarray, prior_C: np.ndarray,
                         threshold: float = 0.5, neg_cutoff: float = -0.4):
    """Fraction of cells with at least one strongly-negative pair active.

    With GR-Neutro's prior (no -1 entries), `neg_cutoff=-0.4` flags pairs like
    (gran_density, texture_unif) = -0.8 and (gran_coarseness, texture_unif) = -0.7.
    Useful as a secondary metric even though hard mutex is undefined.
    """
    pred = (con_probs >= threshold).astype(int)
    K = pred.shape[1]
    neg = (prior_C < neg_cutoff)
    np.fill_diagonal(neg, False)
    if not neg.any():
        return dict(violation_rate=0.0, n_pairs=0)
    pair_violations = []
    per_pair = {}
    for i in range(K):
        for j in range(i + 1, K):
            if neg[i, j]:
                v = pred[:, i] * pred[:, j]
                pair_violations.append(v)
                per_pair[f"{i}-{j}"] = int(v.sum())
    any_v = np.stack(pair_violations).max(axis=0) if pair_violations else np.zeros(len(pred))
    return dict(
        violation_rate=float(any_v.mean()),
        n_pairs=len(pair_violations),
        per_pair_count=per_pair,
        cutoff=neg_cutoff,
    )


# ------------------------------- main ---------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True,
                    help="Directory containing predictions.pt and model.pt.")
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="Conformal target miscoverage level.")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    pred_blob = torch.load(run_dir / "predictions.pt", map_location="cpu", weights_only=False)
    test = pred_blob["test"]
    val = pred_blob["val"]
    is_baseline = bool(pred_blob.get("is_baseline", False))
    class_names = pred_blob["class_names"]
    concepts = pred_blob["concepts"]

    cls_test_p = test["class_logits"].sigmoid().numpy()
    cls_val_p = val["class_logits"].sigmoid().numpy()
    y_test = test["labels"].numpy().astype(int)
    y_val = val["labels"].numpy().astype(int)

    eval_out = {"is_baseline": is_baseline, "class_names": class_names, "concepts": concepts}

    # Per-class best classification threshold (calibrated on val, applied to test).
    from sklearn.metrics import f1_score
    grid = np.linspace(0.05, 0.95, 19)
    best_cls_thr = np.full(y_val.shape[1], 0.5)
    for k in range(y_val.shape[1]):
        if y_val[:, k].sum() == 0:
            continue
        f1s = [f1_score(y_val[:, k], (cls_val_p[:, k] >= t).astype(int), zero_division=0)
               for t in grid]
        best_cls_thr[k] = grid[int(np.argmax(f1s))]
    cls_test_pred_tuned = np.zeros_like(cls_test_p, dtype=int)
    for k in range(cls_test_p.shape[1]):
        cls_test_pred_tuned[:, k] = (cls_test_p[:, k] >= best_cls_thr[k]).astype(int)
    eval_out["classification_tuned"] = dict(
        thresholds=best_cls_thr.tolist(),
        macro_f1=float(f1_score(y_test, cls_test_pred_tuned, average="macro", zero_division=0)),
        weighted_f1=float(f1_score(y_test, cls_test_pred_tuned, average="weighted", zero_division=0)),
        per_class_f1=[float(f1_score(y_test[:, k], cls_test_pred_tuned[:, k], zero_division=0))
                       for k in range(y_test.shape[1])],
    )
    print(f"[tuned class] weighted_f1={eval_out['classification_tuned']['weighted_f1']:.4f}  "
          f"macro_f1={eval_out['classification_tuned']['macro_f1']:.4f}")

    # Per-concept best threshold (only meaningful when concept logits exist).
    if not is_baseline:
        con_val_p = val["concept_logits"].sigmoid().numpy()
        con_test_p = test["concept_logits"].sigmoid().numpy()
        ct_val_hard = (val["concept_targets"].numpy() >= 0.5).astype(int)
        ct_test_hard = (test["concept_targets"].numpy() >= 0.5).astype(int)
        best_con_thr = np.full(con_val_p.shape[1], 0.5)
        for k in range(con_val_p.shape[1]):
            if ct_val_hard[:, k].sum() == 0:
                continue
            f1s = [f1_score(ct_val_hard[:, k], (con_val_p[:, k] >= t).astype(int), zero_division=0)
                   for t in grid]
            best_con_thr[k] = grid[int(np.argmax(f1s))]
        con_test_pred_tuned = (con_test_p >= best_con_thr[None, :]).astype(int)
        per_con_f1 = [float(f1_score(ct_test_hard[:, k], con_test_pred_tuned[:, k], zero_division=0))
                       for k in range(con_test_p.shape[1])]
        eval_out["concept_tuned"] = dict(
            thresholds=best_con_thr.tolist(),
            mean_concept_f1=float(np.mean(per_con_f1)),
            per_concept_f1=per_con_f1,
        )
        print(f"[tuned concept] mean_concept_f1={eval_out['concept_tuned']['mean_concept_f1']:.4f}")

    eval_out["conformal"] = conformal_eval(cls_val_p, y_val, cls_test_p, y_test, alpha=args.alpha)
    print(f"[conformal] alpha={args.alpha}  marginal_pos_cov="
          f"{eval_out['conformal']['marginal_positive_coverage']:.3f}  "
          f"mean_set_size={eval_out['conformal']['mean_set_size']:.2f}")

    if not is_baseline:
        # Completeness probe — fit on train concept logits if available, else
        # fall back to val. train.py dumps train predictions when run with the
        # latest code; older runs only have val/test.
        if "train" in pred_blob:
            con_train = pred_blob["train"]["concept_logits"].numpy()
            y_train = pred_blob["train"]["labels"].numpy().astype(int)
            con_test = test["concept_logits"].numpy()
            eval_out["completeness_probe"] = completeness_probe(
                con_train, y_train, con_test, y_test)
            eval_out["completeness_probe"]["fit_split"] = "train"
        else:
            con_val = val["concept_logits"].numpy()
            con_test = test["concept_logits"].numpy()
            eval_out["completeness_probe"] = completeness_probe(
                con_val, y_val, con_test, y_test)
            eval_out["completeness_probe"]["fit_split"] = "val"
        print(f"[probe] subset_acc={eval_out['completeness_probe']['subset_accuracy']:.3f}  "
              f"weighted_f1={eval_out['completeness_probe']['weighted_f1']:.3f}")

        # Cooccurrence and violation. Load prior C from model.pt if present
        # (single-run case), else fall back to the project concept_config.
        if (run_dir / "model.pt").exists():
            ckpt = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=False)
            prior_C = ckpt["prior_C"].numpy()
        else:
            from models import build_prior_C
            cfg_path = Path(__file__).resolve().parent / "concept_config_gr_neutro.json"
            cfg = json.loads(cfg_path.read_text())
            prior_C = build_prior_C(cfg["concepts"], cfg["concept_constraint_matrix"]).numpy()
        con_test_p = torch.sigmoid(torch.tensor(con_test)).numpy()
        eval_out["cooccurrence"] = cooccurrence_vs_prior(con_test_p, prior_C)
        eval_out["violation"] = violation_rate_soft(con_test_p, prior_C)
        print(f"[cooc] off_diag_pearson(emp,prior)={eval_out['cooccurrence']['off_diag_pearson_corr']:.3f}")
        print(f"[viol] rate={eval_out['violation']['violation_rate']:.3f}  "
              f"pairs={eval_out['violation']['n_pairs']}")

    (run_dir / "eval.json").write_text(json.dumps(eval_out, indent=2))
    print(f"[save] {run_dir}/eval.json")


if __name__ == "__main__":
    main()
