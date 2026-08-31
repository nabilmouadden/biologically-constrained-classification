"""LaBo (Yang et al., CVPR 2023, ``Language in a Bottle'') baseline on GR-Neutro.

LaBo in brief
-------------
1. An LLM (or a fixed prompt template) generates a concept vocabulary per class.
   We skip the LLM-generation step: the GR-Neutro textbook 11-concept rubric is
   already the canonical vocabulary, shared across all baselines.
2. For each concept k, compute its CLIP text embedding t_k.
3. For each cell x_c, compute its CLIP image embedding e_c.
4. Concept score for cell c on concept k is s_{c,k} = cosine(e_c, t_k).
   We use the paired pos/neg variant
       s_{c,k} = cos(e_c, t_k^+) - cos(e_c, t_k^-)
   as is standard for medical VLM grounding and matches our existing
   BiomedCLIP `scores_cosdiff` channel.
5. Train a sparse linear classifier on the (N, M) concept-score matrix
   with class-wise top-K concept pruning (LaBo's signature step).

Difference from our Label-free CBM (Oikarinen 2023) implementation
------------------------------------------------------------------
- We already ran Label-free CBM with per-class L1/L2 logistic on the same
  concept-score matrix (`label_free_cbm.py`).
- LaBo's signature step is the **top-K concept selection per class**: only
  the K=8 strongest concepts (out of 11) survive into the final classifier
  for each of the 7 classes.

Pipeline
--------
1. Load BiomedCLIP scores_v2 (4378 x 11) and labels (4378 x 7).
2. Recreate the canonical 80/10/10 stratified-multilabel split (seed 2024)
   used by all other GR-Neutro baselines.
3. Pre-selection pass: for each class k, fit an L1 logistic on (X_tr, y_tr[:,k])
   with C tuned on validation. Rank the 11 concepts by |coefficient| and
   retain the top-K (default K=8). This is the per-class concept dictionary.
4. Final pass: for each class k, refit an L1 logistic restricted to the
   selected top-K concepts. Tune C and threshold on validation.
5. Report per-class F1, weighted F1, macro F1, the per-class concept
   selection (which K of M concepts each class uses) and a comparison
   row against Label-free CBM.

Outputs
-------
- outputs/labo/results.json
- outputs/labo/concept_selection_per_class.pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Split reconstruction (identical to data.py::stratified_multilabel_split and
# label_free_cbm.py, so the held-out test split matches every other baseline)
# ---------------------------------------------------------------------------

def stratified_multilabel_split(labels_np: np.ndarray, test_size: float = 0.10,
                                 val_size: float = 0.10, seed: int = 2024):
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
# Per-class L1 logistic regression with C and threshold tuned on validation
# ---------------------------------------------------------------------------

def _fit_one_class_l1(X_tr, y_tr_c, X_va, y_va_c, C_grid):
    """Fit L1 logistic for one binary class with C and threshold tuned on val.
    Returns (best_f1_va, C_star, th_star, clf, coef_full).
    coef_full has length X_tr.shape[1]."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score

    best = (-1.0, None, 0.5, None, None)
    for C in C_grid:
        clf = LogisticRegression(penalty="l1", solver="liblinear",
                                  C=C, class_weight="balanced", max_iter=2000)
        clf.fit(X_tr, y_tr_c)
        prob_va = clf.predict_proba(X_va)[:, 1]
        best_th = 0.5
        best_th_f1 = -1.0
        for th in np.linspace(0.1, 0.9, 17):
            f1 = f1_score(y_va_c, (prob_va >= th).astype(int), zero_division=0)
            if f1 > best_th_f1:
                best_th_f1 = f1
                best_th = float(th)
        if best_th_f1 > best[0]:
            best = (best_th_f1, float(C), best_th, clf, clf.coef_.ravel().copy())
    return best


def labo_pipeline(X_tr, y_tr, X_va, y_va, X_te, y_te, class_names,
                  concept_names, K_top: int):
    """Run LaBo on the concept-score matrix.

    Step A: per-class L1 logistic on the full M-dim concept vector to rank
            concepts by |coefficient|. The class-specific top-K survives.
    Step B: per-class L1 logistic refit on the selected K-dim sub-matrix.

    Returns dict with per-class F1, selection, weights, and the value of K.
    """
    from sklearn.metrics import f1_score

    C_grid = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    K_classes = y_tr.shape[1]
    M = X_tr.shape[1]
    K_top = int(min(K_top, M))

    per_class_f1 = []
    per_class_C = []
    per_class_th = []
    per_class_selection = []  # list of concept-index lists (length K_top)
    W_full = np.zeros((K_classes, M), dtype=np.float64)        # ranking coefs
    W_final = np.zeros((K_classes, M), dtype=np.float64)       # refit coefs (sparse: only selected slots populated)

    for c in range(K_classes):
        y_tr_c = y_tr[:, c].astype(int)
        y_va_c = y_va[:, c].astype(int)
        y_te_c = y_te[:, c].astype(int)
        if y_tr_c.sum() == 0 or y_tr_c.sum() == len(y_tr_c):
            per_class_f1.append(0.0)
            per_class_C.append(None)
            per_class_th.append(0.5)
            per_class_selection.append(list(range(K_top)))
            continue

        # --- Step A: rank concepts on the full M-dim matrix ---
        _, _, _, _, coef_full = _fit_one_class_l1(
            X_tr, y_tr_c, X_va, y_va_c, C_grid)
        W_full[c] = coef_full

        # Top-K by |coefficient|. Tie-break preserves concept ordering, so
        # ties (e.g. all-zero coefs in extreme regularisation) still produce
        # a deterministic selection.
        order = np.argsort(-np.abs(coef_full), kind="stable")
        sel = sorted(order[:K_top].tolist())
        per_class_selection.append(sel)

        # --- Step B: refit on the K-dim sub-matrix ---
        X_tr_s = X_tr[:, sel]
        X_va_s = X_va[:, sel]
        X_te_s = X_te[:, sel]
        f1_va, C_star, th_star, clf, coef_sel = _fit_one_class_l1(
            X_tr_s, y_tr_c, X_va_s, y_va_c, C_grid)

        prob_te = clf.predict_proba(X_te_s)[:, 1]
        pred_te = (prob_te >= th_star).astype(int)
        f1_te = f1_score(y_te_c, pred_te, zero_division=0)

        per_class_f1.append(float(f1_te))
        per_class_C.append(float(C_star))
        per_class_th.append(float(th_star))
        # Scatter the sub-matrix coefficients back into the full M-dim slot
        for j, idx in enumerate(sel):
            W_final[c, idx] = float(coef_sel[j])

    return {
        "per_class_f1": per_class_f1,
        "per_class_C": per_class_C,
        "per_class_threshold": per_class_th,
        "per_class_selection_idx": per_class_selection,
        "per_class_selection_names": [
            [concept_names[i] for i in sel] for sel in per_class_selection
        ],
        "K_top": K_top,
        "W_ranking": W_full.tolist(),
        "W_final": W_final.tolist(),
    }


def macro_f1(per_class):
    return float(np.asarray(per_class, dtype=float).mean())


def weighted_f1_from_per_class(per_class, y_te):
    support = y_te.sum(axis=0).astype(float)
    if support.sum() == 0:
        return 0.0
    w = support / support.sum()
    return float((np.asarray(per_class) * w).sum())


# ---------------------------------------------------------------------------
# Concept-selection figure: binary heatmap of per-class K-of-M selection
# overlaid with the refit logistic coefficients.
# ---------------------------------------------------------------------------

def plot_concept_selection(W_final, selection_mask, class_names, concept_names,
                           K_top, out_pdf: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vmax = float(np.abs(W_final).max()) if W_final.size else 1.0
    vmax = max(vmax, 1e-6)

    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    im = ax.imshow(W_final, aspect="auto", cmap="RdBu_r",
                    vmin=-vmax, vmax=vmax)

    # Outline selected cells (top-K per class) with a black box
    for c in range(W_final.shape[0]):
        for j in range(W_final.shape[1]):
            if selection_mask[c, j]:
                ax.add_patch(plt.Rectangle((j - 0.5, c - 0.5), 1, 1,
                                             fill=False, edgecolor="black",
                                             linewidth=1.3))
            if abs(W_final[c, j]) > 1e-6:
                ax.text(j, c, f"{W_final[c, j]:.2f}", ha="center",
                        va="center", fontsize=7, color="black")

    ax.set_xticks(range(len(concept_names)))
    ax.set_xticklabels(concept_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_title(f"LaBo per-class top-K selection (K={K_top}, M={len(concept_names)})")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("logistic-regression coefficient (refit on selected concepts)")
    plt.tight_layout()
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Comparison row vs. Label-free CBM on the same fixed test split
# ---------------------------------------------------------------------------

def load_lfcbm(lfcbm_path: Path):
    if not lfcbm_path.exists():
        return None
    j = json.loads(lfcbm_path.read_text())
    return {
        "L1": {
            "per_class_f1": j["L1"]["per_class_f1"],
            "macro_f1": j["L1"]["macro_f1"],
            "weighted_f1": j["L1"]["weighted_f1"],
            "sparsity_per_class": j["L1"].get("sparsity_per_class"),
        },
        "L2": {
            "per_class_f1": j["L2"]["per_class_f1"],
            "macro_f1": j["L2"]["macro_f1"],
            "weighted_f1": j["L2"]["weighted_f1"],
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=str,
                    default="outputs/vlm_grounding/scores_v2/biomedclip/scores.npz",
                    help="BiomedCLIP scores_v2 npz; expects keys "
                         "'scores' (raw cos sim, N x 11), "
                         "'scores_cosdiff' (paired pos-neg diff, N x 11), "
                         "'labels' (N x 7), 'concepts' (11,), 'class_names' (7,).")
    ap.add_argument("--output", type=str, default="outputs/labo")
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--K_top", type=int, default=8,
                    help="Top-K concepts retained per class (LaBo signature step). "
                         "Defaults to 8 of 11 as briefed.")
    ap.add_argument("--score_key", type=str, default="scores_cosdiff",
                    help="Which BiomedCLIP score variant to use as concept signal. "
                         "Defaults to the paired pos/neg cosine-diff channel, which is "
                         "the LaBo-style scoring on our setup.")
    ap.add_argument("--lfcbm_results", type=str,
                    default="outputs/label_free_cbm/results.json",
                    help="Label-free CBM results.json for the comparison row.")
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load BiomedCLIP scores_v2 + labels
    npz = np.load(args.scores, allow_pickle=True)
    if args.score_key not in npz.files:
        raise SystemExit(
            f"score_key={args.score_key!r} not in {args.scores}; "
            f"available: {list(npz.files)}")
    X = npz[args.score_key].astype(np.float32)
    y = npz["labels"].astype(np.int64)
    concept_names = [str(c) for c in npz["concepts"]]
    class_names = [str(c) for c in npz["class_names"]]

    N, M = X.shape
    K_classes = y.shape[1]
    print(f"[labo] X={X.shape} ({args.score_key}), y={y.shape}, "
          f"M concepts={M}, K classes={K_classes}, K_top={args.K_top}")

    # ---- Canonical 80/10/10 split
    tr_idx, va_idx, te_idx = stratified_multilabel_split(
        y, test_size=0.10, val_size=0.10, seed=args.seed)
    print(f"[labo] split: n_train={len(tr_idx)} n_val={len(va_idx)} "
          f"n_test={len(te_idx)}")

    X_tr, X_va, X_te = X[tr_idx], X[va_idx], X[te_idx]
    y_tr, y_va, y_te = y[tr_idx], y[va_idx], y[te_idx]

    # ---- Run LaBo
    print(f"[labo] fitting per-class L1 with top-K={args.K_top} pruning ...")
    labo = labo_pipeline(X_tr, y_tr, X_va, y_va, X_te, y_te,
                          class_names, concept_names, K_top=args.K_top)

    pcf = labo["per_class_f1"]
    mF1 = macro_f1(pcf)
    wF1 = weighted_f1_from_per_class(pcf, y_te)
    print(f"[labo] per-class F1: {[round(f, 3) for f in pcf]}")
    print(f"[labo] macro-F1 = {mF1:.4f}  weighted-F1 = {wF1:.4f}")
    print("[labo] per-class concept selection:")
    for c, sel in enumerate(labo["per_class_selection_names"]):
        print(f"  {class_names[c]:18s}: {sel}")

    # Selection frequency across classes (which concepts are picked most often)
    selection_mask = np.zeros((K_classes, M), dtype=int)
    for c, sel in enumerate(labo["per_class_selection_idx"]):
        for j in sel:
            selection_mask[c, j] = 1
    concept_pick_count = selection_mask.sum(axis=0).tolist()
    pick_pattern = sorted(
        zip(concept_names, concept_pick_count),
        key=lambda t: (-t[1], t[0]),
    )
    print("[labo] concept pick frequency (out of K classes):")
    for n, c in pick_pattern:
        print(f"  {n:36s}  picked by {c}/{K_classes}")

    # ---- Comparison row vs. Label-free CBM on the same test split
    lfcbm = load_lfcbm(Path(args.lfcbm_results))

    # ---- Persist
    results = {
        "config": {
            "scores_path": args.scores,
            "score_key": args.score_key,
            "seed": args.seed,
            "K_top": args.K_top,
            "n_total": int(N),
            "n_train": int(len(tr_idx)),
            "n_val": int(len(va_idx)),
            "n_test": int(len(te_idx)),
            "n_concepts": int(M),
            "n_classes": int(K_classes),
        },
        "class_names": class_names,
        "concept_names": concept_names,
        "labo": {
            "per_class_f1": pcf,
            "macro_f1": mF1,
            "weighted_f1": wF1,
            "best_C_per_class": labo["per_class_C"],
            "best_threshold_per_class": labo["per_class_threshold"],
            "per_class_selection_idx": labo["per_class_selection_idx"],
            "per_class_selection_names": labo["per_class_selection_names"],
            "concept_pick_count": concept_pick_count,
            "W_ranking": labo["W_ranking"],
            "W_final": labo["W_final"],
        },
        "comparison_label_free_cbm": lfcbm,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[labo] wrote {out_dir/'results.json'}")

    # ---- Selection figure
    W_final = np.asarray(labo["W_final"], dtype=np.float64)
    plot_concept_selection(W_final, selection_mask, class_names, concept_names,
                            K_top=args.K_top,
                            out_pdf=out_dir / "concept_selection_per_class.pdf")
    print(f"[labo] wrote {out_dir/'concept_selection_per_class.pdf'}")

    # ---- Console summary
    print("\n=== LaBo summary ===")
    print(f"LaBo: macro-F1 = {mF1:.4f}  weighted-F1 = {wF1:.4f}")
    if lfcbm is not None:
        print(f"Label-free CBM L1: macro-F1 = {lfcbm['L1']['macro_f1']:.4f}  "
              f"weighted-F1 = {lfcbm['L1']['weighted_f1']:.4f}")
        print(f"Label-free CBM L2: macro-F1 = {lfcbm['L2']['macro_f1']:.4f}  "
              f"weighted-F1 = {lfcbm['L2']['weighted_f1']:.4f}")
        dW = wF1 - lfcbm["L1"]["weighted_f1"]
        dM = mF1 - lfcbm["L1"]["macro_f1"]
        sign = "helps" if dW > 0 else "hurts"
        print(f"Top-K pruning {sign} vs Label-free CBM L1: "
              f"dW-F1={dW:+.4f}, dM-F1={dM:+.4f}")
    else:
        print("[labo] Label-free CBM results not found; comparison row skipped.")


if __name__ == "__main__":
    main()
