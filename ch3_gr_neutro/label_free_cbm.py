"""Label-free Concept Bottleneck Model baseline (Oikarinen et al., ICLR 2023).

Implements the Label-free CBM idea on GR-Neutro: a sparse linear classifier
fitted on top of frozen CLIP image-text similarity concept scores, with no
per-cell concept labels used during training.

Pipeline
--------
1. Load BiomedCLIP per-cell concept similarity scores  (N=4378, K=11).
   These were precomputed in the VL-grounding stage as cos(image_emb,
   text_emb("a photo of a cell with {concept}")).
2. Recreate the same 80/10/10 stratified multilabel split used by all
   other GR-Neutro baselines (seed 2024 by default).
3. For each of the 7 classes, fit a per-class logistic regression on the
   11-dim concept score vector. Two variants:
     - L1-regularised (sparse), C tuned on the validation set.
     - L2-regularised (dense), C tuned on the validation set.
4. Report per-class F1 on the held-out test set, sparsity (count of
   non-zero concept weights per class for the L1 fit), and append a
   side-by-side comparison row against the existing baselines.

Outputs (under --output, default outputs/label_free_cbm/):
- results.json              # F1 + sparsity + comparison
- concept_weights_per_class.pdf  # heatmap of L1 coefficients per class
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Split reconstruction (matches data.py::stratified_multilabel_split)
# ---------------------------------------------------------------------------

def stratified_multilabel_split(labels_np: np.ndarray, test_size: float = 0.10,
                                 val_size: float = 0.10, seed: int = 2024):
    """Replicates ch3_gr_neutro/data.py exactly so the held-out test split
    is identical to the one used by all other GR-Neutro baselines."""
    from sklearn.model_selection import StratifiedShuffleSplit
    K = labels_np.shape[1]
    class_count = labels_np.sum(axis=0).astype(float)
    class_count[class_count == 0] = labels_np.shape[0]
    strat = np.empty(len(labels_np), dtype=int)
    for i in range(len(labels_np)):
        active = np.where(labels_np[i] == 1)[0]
        strat[i] = int(active[np.argmin(class_count[active])]) if len(active) else -1

    idx_all = np.arange(len(labels_np))
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(sss1.split(idx_all, strat))
    rel = val_size / (1.0 - test_size)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=rel, random_state=seed)
    sub_train, sub_val = next(sss2.split(train_val_idx, strat[train_val_idx]))
    return train_val_idx[sub_train], train_val_idx[sub_val], test_idx


# ---------------------------------------------------------------------------
# Per-class L1 / L2 logistic regression with C tuned on validation set
# ---------------------------------------------------------------------------

def fit_per_class_logreg(X_tr, y_tr, X_va, y_va, X_te, y_te, penalty: str,
                          class_names, n_concepts: int):
    """Fit one LR per class. Returns (per_class_f1, weights[K_classes,K_concepts])."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score

    C_grid = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    K_classes = y_tr.shape[1]
    per_class_f1 = []
    per_class_C = []
    per_class_th = []
    W = np.zeros((K_classes, n_concepts), dtype=np.float64)

    for c in range(K_classes):
        y_tr_c = y_tr[:, c].astype(int)
        y_va_c = y_va[:, c].astype(int)
        y_te_c = y_te[:, c].astype(int)
        # If class is empty in train, skip
        if y_tr_c.sum() == 0 or y_tr_c.sum() == len(y_tr_c):
            per_class_f1.append(0.0)
            per_class_C.append(None)
            per_class_th.append(0.5)
            continue

        # Tune C on validation F1 (with default threshold 0.5)
        best = (-1.0, None, 0.5, None)  # (f1_va, C, threshold, model)
        for C in C_grid:
            clf = LogisticRegression(penalty=penalty, solver="liblinear",
                                      C=C, class_weight="balanced",
                                      max_iter=2000)
            clf.fit(X_tr, y_tr_c)
            prob_va = clf.predict_proba(X_va)[:, 1]
            # Also tune threshold on val (coarse grid)
            best_th = 0.5
            best_th_f1 = -1.0
            for th in np.linspace(0.1, 0.9, 17):
                f1 = f1_score(y_va_c, (prob_va >= th).astype(int),
                              zero_division=0)
                if f1 > best_th_f1:
                    best_th_f1 = f1
                    best_th = float(th)
            if best_th_f1 > best[0]:
                best = (best_th_f1, C, best_th, clf)

        _, C_star, th_star, clf = best
        prob_te = clf.predict_proba(X_te)[:, 1]
        pred_te = (prob_te >= th_star).astype(int)
        f1_te = f1_score(y_te_c, pred_te, zero_division=0)
        per_class_f1.append(float(f1_te))
        per_class_C.append(float(C_star))
        per_class_th.append(float(th_star))
        W[c] = clf.coef_.ravel()

    return per_class_f1, per_class_C, per_class_th, W


def macro_f1(per_class):
    arr = np.asarray(per_class, dtype=float)
    return float(arr.mean())


def weighted_f1_from_per_class(per_class, y_te):
    support = y_te.sum(axis=0).astype(float)
    if support.sum() == 0:
        return 0.0
    w = support / support.sum()
    return float((np.asarray(per_class) * w).sum())


# ---------------------------------------------------------------------------
# Comparison row vs. existing baselines (Vanilla / Joint CBM / CBM-best / LG)
# ---------------------------------------------------------------------------

def load_baseline_row(eval_path: Path):
    if not eval_path.exists():
        return None
    j = json.loads(eval_path.read_text())
    tuned = j.get("classification_tuned") or {}
    return {
        "macro_f1": tuned.get("macro_f1"),
        "weighted_f1": tuned.get("weighted_f1"),
        "per_class_f1": tuned.get("per_class_f1"),
    }


def assemble_comparison(outputs_root: Path, results: dict):
    """Pull the headline numbers from sibling baselines for the same test
    split. Mapping (best-effort, falls back gracefully if a run is absent):
       Vanilla              -> B_kitchen_s2024              (no concepts)
       Joint concept-sup    -> cbm_lcon1p0_s42              (joint CBM)
       CBM-best             -> clean_v10_lc01_s2024         (current best)
       Language-grounded    -> ls_lam0p3_s2024              (LG variant)
    """
    candidates = {
        "vanilla": ["B_kitchen_s2024"],
        "joint_concept_supervised": ["cbm_lcon1p0_s42", "cbm_lcon0p5_s42"],
        "cbm_best": ["clean_v10_lc01_s2024", "clean_v14_kitchen_s2024",
                     "clean_v8_unfreeze9_s2024"],
        "language_grounded": ["ls_lam0p3_s2024", "ls_lam1p0_s2024",
                               "ls_lam0p05_s2024"],
    }
    table = {}
    for name, runs in candidates.items():
        for r in runs:
            row = load_baseline_row(outputs_root / r / "eval.json")
            if row is not None:
                table[name] = {"run": r, **row}
                break
        else:
            table[name] = None

    table["label_free_cbm_L1"] = {
        "run": "label_free_cbm",
        "macro_f1": results["L1"]["macro_f1"],
        "weighted_f1": results["L1"]["weighted_f1"],
        "per_class_f1": results["L1"]["per_class_f1"],
    }
    table["label_free_cbm_L2"] = {
        "run": "label_free_cbm",
        "macro_f1": results["L2"]["macro_f1"],
        "weighted_f1": results["L2"]["weighted_f1"],
        "per_class_f1": results["L2"]["per_class_f1"],
    }
    return table


# ---------------------------------------------------------------------------
# Concept-weight heatmap
# ---------------------------------------------------------------------------

def plot_concept_weights(W: np.ndarray, class_names, concept_names, out_pdf: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Symmetric color scale around zero for sign visibility
    vmax = float(np.abs(W).max()) if W.size else 1.0
    vmax = max(vmax, 1e-6)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    im = ax.imshow(W, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(concept_names)))
    ax.set_xticklabels(concept_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_title("Label-free CBM (L1) per-class concept weights")
    # Annotate non-zero cells
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            if abs(W[i, j]) > 1e-6:
                ax.text(j, i, f"{W[i, j]:.2f}", ha="center", va="center",
                        fontsize=7, color="black")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("logistic-regression coefficient")
    plt.tight_layout()
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=str,
                    default="outputs/vlm_grounding/scores/biomedclip_scores.npz",
                    help="npz with keys v1_scores (N,11) and labels (N,7)")
    ap.add_argument("--output", type=str, default="outputs/label_free_cbm")
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--score_key", type=str, default="v1_scores",
                    choices=["v1_scores", "v2_scores"],
                    help="Which BiomedCLIP score variant to use as concept "
                         "signal (v1 = raw cos sim, v2 = paired diff).")
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load BiomedCLIP scores + labels
    npz = np.load(args.scores, allow_pickle=True)
    if "v1_scores" in npz.files:
        X = npz[args.score_key].astype(np.float32)   # (N, 11)
        y = npz["labels"].astype(np.int64)           # (N, 7)
        concept_names = [str(c) for c in npz["concepts"]]
        class_names = [str(c) for c in npz["class_names"]]
    else:
        # Fallback for the simpler dump
        X = npz["biomed_scores"].astype(np.float32)
        y = npz["labels"].astype(np.int64)
        concept_names = [str(c) for c in npz["concepts"]]
        class_names = [str(c) for c in npz["class_names"]]

    N, K_concepts = X.shape
    K_classes = y.shape[1]
    print(f"[lfcbm] X={X.shape}, y={y.shape}, "
          f"K_concepts={K_concepts}, K_classes={K_classes}")

    # ---- Reconstruct the canonical 80/10/10 split (seed 2024)
    tr_idx, va_idx, te_idx = stratified_multilabel_split(
        y, test_size=0.10, val_size=0.10, seed=args.seed)
    print(f"[lfcbm] split: n_train={len(tr_idx)} n_val={len(va_idx)} "
          f"n_test={len(te_idx)}")

    X_tr, X_va, X_te = X[tr_idx], X[va_idx], X[te_idx]
    y_tr, y_va, y_te = y[tr_idx], y[va_idx], y[te_idx]

    # ---- L1 fit (sparse Label-free CBM)
    print("[lfcbm] fitting L1 per-class logistic regression ...")
    pcf_l1, C_l1, th_l1, W_l1 = fit_per_class_logreg(
        X_tr, y_tr, X_va, y_va, X_te, y_te,
        penalty="l1", class_names=class_names, n_concepts=K_concepts)
    sparsity = [int((np.abs(W_l1[c]) > 1e-6).sum()) for c in range(K_classes)]
    print(f"[lfcbm] L1 per-class F1: {[round(f, 3) for f in pcf_l1]}")
    print(f"[lfcbm] L1 non-zero concepts per class: {sparsity}")

    # ---- L2 fit (dense logistic-regression baseline on the same features)
    print("[lfcbm] fitting L2 per-class logistic regression ...")
    pcf_l2, C_l2, th_l2, W_l2 = fit_per_class_logreg(
        X_tr, y_tr, X_va, y_va, X_te, y_te,
        penalty="l2", class_names=class_names, n_concepts=K_concepts)
    print(f"[lfcbm] L2 per-class F1: {[round(f, 3) for f in pcf_l2]}")

    # ---- Assemble results
    results = {
        "config": {
            "scores_path": args.scores,
            "score_key": args.score_key,
            "seed": args.seed,
            "n_total": int(N),
            "n_train": int(len(tr_idx)),
            "n_val": int(len(va_idx)),
            "n_test": int(len(te_idx)),
            "n_concepts": int(K_concepts),
            "n_classes": int(K_classes),
        },
        "class_names": class_names,
        "concept_names": concept_names,
        "L1": {
            "per_class_f1": pcf_l1,
            "macro_f1": macro_f1(pcf_l1),
            "weighted_f1": weighted_f1_from_per_class(pcf_l1, y_te),
            "best_C_per_class": C_l1,
            "best_threshold_per_class": th_l1,
            "sparsity_per_class": sparsity,
            "weights": W_l1.tolist(),
        },
        "L2": {
            "per_class_f1": pcf_l2,
            "macro_f1": macro_f1(pcf_l2),
            "weighted_f1": weighted_f1_from_per_class(pcf_l2, y_te),
            "best_C_per_class": C_l2,
            "best_threshold_per_class": th_l2,
            "weights": W_l2.tolist(),
        },
    }

    # ---- Comparison row vs. other baselines
    outputs_root = Path(args.scores).resolve().parent.parent  # outputs/
    if outputs_root.name != "outputs":
        # Fallback if user passed a non-standard path
        outputs_root = Path("outputs")
    results["comparison"] = assemble_comparison(outputs_root, results)

    # ---- Persist
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[lfcbm] wrote {out_dir/'results.json'}")

    plot_concept_weights(W_l1, class_names, concept_names,
                         out_dir / "concept_weights_per_class.pdf")
    print(f"[lfcbm] wrote {out_dir/'concept_weights_per_class.pdf'}")

    # ---- Console summary
    print("\n=== Label-free CBM summary ===")
    print(f"L1: macro-F1 = {results['L1']['macro_f1']:.4f}  "
          f"weighted-F1 = {results['L1']['weighted_f1']:.4f}")
    print(f"L2: macro-F1 = {results['L2']['macro_f1']:.4f}  "
          f"weighted-F1 = {results['L2']['weighted_f1']:.4f}")
    print("comparison (weighted_f1 / macro_f1):")
    for k, v in results["comparison"].items():
        if v is None:
            print(f"  {k:30s}  MISSING")
        else:
            w = v.get("weighted_f1")
            m = v.get("macro_f1")
            ws = "n/a" if w is None else f"{w:.4f}"
            ms = "n/a" if m is None else f"{m:.4f}"
            print(f"  {k:30s}  W-F1={ws}  M-F1={ms}  ({v.get('run')})")


if __name__ == "__main__":
    main()
