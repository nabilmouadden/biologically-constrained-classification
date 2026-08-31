"""TCAV baseline for GR-Neutro (Kim et al., 2018).

Probes a trained vanilla classifier (no concept supervision) for the
directional sensitivity of each class logit to each of the 11 GR-Neutro
concepts. Concepts have no per-cell binary labels, so we derive
positive/negative sets from the docx class-conditional prior matrix
(7 classes x 11 concepts in concept_config_gr_neutro.json). For cell i
and concept k:

    t_hat[i, k] = max_{c : y[i, c] = 1}  C[c, k]            (graded prior)

with positives  t_hat >= 0.6  and  negatives  t_hat <= 0.4.

Penultimate activation = LayerNorm(CLS) for the dinobloom_b backbone, i.e.
the input to the final Linear layer of the classifier head saved in
outputs/vanilla_s42/model.pt. The classifier head is
  Sequential(LayerNorm(768), Dropout, Linear(768 -> 7))
so the CAV space is R^768.

This script is run on Ruche by jobs/tcav_job.sh.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import binomtest
from sklearn.svm import LinearSVC


def load_concept_config(path: Path):
    with open(path) as f:
        cfg = json.load(f)
    concepts = cfg["concepts"]
    classes = cfg["gr_neutro_classes"]
    C = np.asarray(cfg["class_to_concept_matrix"]["matrix"], dtype=np.float32)
    assert C.shape == (len(classes), len(concepts)), f"C shape mismatch: {C.shape}"
    return concepts, classes, C


def load_annotations(csv_path: Path, classes: list[str]):
    """Returns {basename: one-hot label (np.float32, K)}."""
    out = {}
    with open(csv_path) as f:
        r = csv.reader(f)
        header = next(r)
        col_names = header[2:]
        assert col_names == classes, f"CSV class order mismatch: {col_names} vs {classes}"
        for row in r:
            fn = row[0]
            lab = np.asarray([int(x) for x in row[2:]], dtype=np.float32)
            out[fn] = lab
    return out


def load_features(npz_path: Path):
    d = np.load(npz_path, allow_pickle=True)
    paths = np.asarray(d["paths"])
    feats = np.asarray(d["features"], dtype=np.float32)
    basenames = np.asarray([Path(p).name for p in paths])
    return basenames, feats


class VanillaHead(nn.Module):
    """LayerNorm -> Dropout(eval=identity) -> Linear, copied from JointModel."""

    def __init__(self, d: int, num_classes: int):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.fc = nn.Linear(d, num_classes)

    def load_from_state_dict(self, sd: dict):
        self.ln.weight.data.copy_(sd["classifier.0.weight"])
        self.ln.bias.data.copy_(sd["classifier.0.bias"])
        self.fc.weight.data.copy_(sd["classifier.2.weight"])
        self.fc.bias.data.copy_(sd["classifier.2.bias"])

    def penultimate(self, cls_feat: torch.Tensor) -> torch.Tensor:
        """LayerNorm(CLS). This is the activation TCAV probes."""
        return self.ln(cls_feat)

    def logits_from_penult(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc(h)


def graded_prior(labels: np.ndarray, C: np.ndarray) -> np.ndarray:
    """For each cell i and concept k, t_hat[i, k] = max_c y[i,c] * C[c, k].

    Cells with no active class get prior = 0 for every concept.
    """
    # labels: (N, n_class)  C: (n_class, n_concept)
    # We need a per-cell, per-concept max over active classes.
    Y = labels[:, :, None]           # (N, n_class, 1)
    M = Y * C[None, :, :]            # (N, n_class, n_concept)
    M = np.where(Y > 0, M, -np.inf)
    t = M.max(axis=1)                # (N, n_concept)
    t[np.isneginf(t)] = 0.0
    return t.astype(np.float32)


def fit_cav(H_pos: np.ndarray, H_neg: np.ndarray, seed: int):
    """Linear SVM normal direction as CAV (unit vector)."""
    X = np.concatenate([H_pos, H_neg], axis=0)
    y = np.concatenate([np.ones(len(H_pos)), np.zeros(len(H_neg))]).astype(np.int32)
    clf = LinearSVC(C=0.1, max_iter=5000, dual="auto", random_state=seed)
    clf.fit(X, y)
    w = clf.coef_.reshape(-1).astype(np.float32)
    n = np.linalg.norm(w)
    if n < 1e-9:
        return None
    return w / n


def tcav_score_for_cav(head: VanillaHead, H_test: torch.Tensor, cav: np.ndarray,
                       class_idx: int) -> float:
    """Fraction of test cells with positive directional derivative of
    logit[class_idx] along the CAV.
    """
    cav_t = torch.from_numpy(cav).to(H_test.device).float()
    H = H_test.detach().clone().requires_grad_(True)
    logits = head.fc(H)
    target = logits[:, class_idx].sum()
    grads, = torch.autograd.grad(target, H)
    # grads: (N, d). Each row is d logit_k / d h_i for cell i.
    dot = grads @ cav_t
    return float((dot > 0).float().mean().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classifier", required=True,
                    help="Path to vanilla checkpoint (model.pt).")
    ap.add_argument("--features",
                    default="outputs/dinobloom_features.npz",
                    help="Cached DinoBloom-B CLS features.")
    ap.add_argument("--config", default="concept_config_gr_neutro.json")
    ap.add_argument("--annotations",
                    default="/gpfs/workdir/mouaddenn/data/gr_neutro_extended/annotations.csv")
    ap.add_argument("--output", required=True, help="Output directory.")
    ap.add_argument("--bootstrap", type=int, default=50)
    ap.add_argument("--pos_threshold", type=float, default=0.6)
    ap.add_argument("--neg_threshold", type=float, default=0.4)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # ---- Config / labels / features ---------------------------------------
    concepts, classes, C = load_concept_config(Path(args.config))
    print(f"[tcav] {len(concepts)} concepts x {len(classes)} classes")

    annots = load_annotations(Path(args.annotations), classes)
    basenames, feats = load_features(Path(args.features))
    keep = np.asarray([b in annots for b in basenames])
    basenames = basenames[keep]
    feats = feats[keep]
    Y = np.stack([annots[b] for b in basenames], axis=0).astype(np.float32)
    print(f"[tcav] matched {len(basenames)} / {len(keep)} cells with labels")

    # ---- Classifier head --------------------------------------------------
    ckpt = torch.load(args.classifier, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    d_feat = feats.shape[1]
    n_class = len(classes)
    head = VanillaHead(d_feat, n_class).to(args.device).eval()
    head.load_from_state_dict(sd)

    # ---- Penultimate activations ------------------------------------------
    with torch.no_grad():
        feats_t = torch.from_numpy(feats).to(args.device)
        H_all = head.penultimate(feats_t).cpu().numpy().astype(np.float32)
    print(f"[tcav] activations: {H_all.shape}")

    # ---- Splits (use vanilla_s42 split for held-out test set) -------------
    split_path = Path(args.classifier).parent / "split.json"
    if split_path.exists():
        with open(split_path) as f:
            split = json.load(f)
        # split idxs were over the full annotated set in order of CSV/load.
        # We use them by re-indexing into the matched 'basenames' array.
        # The training pipeline builds rows from CSV order then splits;
        # to be robust we restrict to indices < len(basenames).
        test_idx = np.asarray([i for i in split["test_idx"] if i < len(basenames)],
                              dtype=np.int64)
    else:
        test_idx = np.arange(len(basenames))
    print(f"[tcav] held-out test cells: {len(test_idx)}")

    H_test = torch.from_numpy(H_all[test_idx]).to(args.device)
    Y_test = Y[test_idx]
    test_class = Y_test.argmax(axis=1)
    has_class = Y_test.sum(axis=1) > 0

    # ---- Per-cell graded prior --------------------------------------------
    t_hat = graded_prior(Y, C)              # (N, n_concept)
    # Class-level concept signal: per class, mean prior across train cells in
    # that class. Used to define biologically-expected direction.
    # Build class-mean prior using only single-class cells (cleaner signal).
    single = (Y.sum(axis=1) == 1)
    cls_idx_all = Y.argmax(axis=1)
    class_concept_prior = np.zeros((n_class, len(concepts)), dtype=np.float32)
    for c in range(n_class):
        m = single & (cls_idx_all == c)
        if m.sum() > 0:
            class_concept_prior[c] = t_hat[m].mean(axis=0)
        else:
            class_concept_prior[c] = C[c]   # fallback

    # ---- TCAV loop --------------------------------------------------------
    n_boot = args.bootstrap
    tcav_scores = np.full((len(concepts), n_class), np.nan, dtype=np.float32)
    tcav_pvals = np.full((len(concepts), n_class), np.nan, dtype=np.float32)
    significant = np.zeros((len(concepts), n_class), dtype=bool)
    expected_pos = (class_concept_prior > 0.5)   # (n_class, n_concept) -> .T below

    for k, name in enumerate(concepts):
        pos_mask = t_hat[:, k] >= args.pos_threshold
        neg_mask = t_hat[:, k] <= args.neg_threshold
        # Build pool from non-test cells so the SVM and the TCAV test cells are disjoint.
        non_test = np.ones(len(basenames), dtype=bool)
        non_test[test_idx] = False
        pos_idx = np.where(pos_mask & non_test)[0]
        neg_idx = np.where(neg_mask & non_test)[0]
        n_pos, n_neg = len(pos_idx), len(neg_idx)
        print(f"[tcav] concept {k:2d} {name:38s} pos={n_pos} neg={n_neg}")
        if n_pos < 20 or n_neg < 20:
            print(f"[tcav]   too few samples; skipping concept")
            continue

        boot_scores = np.zeros((n_boot, n_class), dtype=np.float32)
        for b in range(n_boot):
            # Stratified subsample with replacement for bootstrap.
            sub_n = min(n_pos, n_neg)
            pi = rng.choice(pos_idx, size=sub_n, replace=True)
            ni = rng.choice(neg_idx, size=sub_n, replace=True)
            cav = fit_cav(H_all[pi], H_all[ni], seed=int(rng.integers(0, 2**31 - 1)))
            if cav is None:
                boot_scores[b] = 0.5
                continue
            for c in range(n_class):
                cell_mask = has_class & (test_class == c)
                if cell_mask.sum() == 0:
                    boot_scores[b, c] = np.nan
                    continue
                H_sub = H_test[cell_mask]
                boot_scores[b, c] = tcav_score_for_cav(head, H_sub, cav, c)

        # Aggregate: median TCAV across bootstraps.
        med = np.nanmedian(boot_scores, axis=0)
        tcav_scores[k] = med
        # Two-sided sign test: count fraction of bootstraps with score > 0.5
        # vs == 0.5 ignored, against null prob = 0.5.
        for c in range(n_class):
            col = boot_scores[:, c]
            col = col[~np.isnan(col)]
            if len(col) == 0:
                continue
            n_above = int((col > 0.5).sum())
            n_below = int((col < 0.5).sum())
            n_eff = n_above + n_below
            if n_eff == 0:
                p = 1.0
            else:
                p = binomtest(n_above, n_eff, p=0.5, alternative="two-sided").pvalue
            tcav_pvals[k, c] = p
            significant[k, c] = (p < args.alpha)

    # ---- Headline ---------------------------------------------------------
    # Significant AND in biologically-expected direction (sign of score - 0.5
    # matches the sign of class_concept_prior[c, k] - 0.5).
    expected_dir = (class_concept_prior > 0.5).T          # (n_concept, n_class)
    observed_dir = (tcav_scores > 0.5)                    # same shape
    agree = (expected_dir == observed_dir)
    headline_mask = significant & agree
    # Headline fraction is over (concept, class) cells where the docx prior
    # is informative (i.e., expected_dir is well-defined). All 11x7 cells
    # qualify because C has graded values in [0,1].
    total = significant.size
    headline = float(headline_mask.sum()) / float(total)
    print(f"[tcav] HEADLINE: {headline_mask.sum()}/{total} = {headline:.3f}")

    # ---- Persist ----------------------------------------------------------
    results = {
        "concepts": concepts,
        "classes": classes,
        "tcav_scores": tcav_scores.tolist(),
        "bootstrap_pvalues": tcav_pvals.tolist(),
        "significant": significant.astype(int).tolist(),
        "expected_direction_positive": expected_dir.astype(int).tolist(),
        "class_concept_prior": class_concept_prior.tolist(),
        "headline_fraction_significant_in_expected_direction": headline,
        "n_bootstrap": n_boot,
        "alpha": args.alpha,
        "pos_threshold": args.pos_threshold,
        "neg_threshold": args.neg_threshold,
        "n_test_cells": int(len(test_idx)),
        "n_features": int(d_feat),
        "checkpoint": str(Path(args.classifier).resolve()),
    }
    with open(out_dir / "tcav_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[tcav] wrote {out_dir / 'tcav_results.json'}")

    # ---- Heatmap PDF ------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 8.5))
    im = ax.imshow(tcav_scores, vmin=0.0, vmax=1.0, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(n_class))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(range(len(concepts)))
    ax.set_yticklabels(concepts)
    for k in range(len(concepts)):
        for c in range(n_class):
            v = tcav_scores[k, c]
            if np.isnan(v):
                continue
            txt = f"{v:.2f}"
            if significant[k, c]:
                txt += "*"
            ax.text(c, k, txt, ha="center", va="center", fontsize=7,
                    color="black" if 0.25 < v < 0.75 else "white")
    ax.set_title(f"TCAV scores ({n_boot}-bootstrap median, * p<{args.alpha})\n"
                 f"headline={headline:.3f}")
    fig.colorbar(im, ax=ax, label="TCAV score")
    fig.tight_layout()
    fig.savefig(out_dir / "tcav_heatmap.pdf")
    plt.close(fig)
    print(f"[tcav] wrote {out_dir / 'tcav_heatmap.pdf'}")

    # ---- Per-concept bar chart -------------------------------------------
    fig, axes = plt.subplots(len(concepts), 1, figsize=(7.0, 1.4 * len(concepts)),
                             sharex=True)
    for k, ax in enumerate(axes):
        vals = tcav_scores[k]
        sig = significant[k]
        colors = ["#1f77b4" if s else "#bbbbbb" for s in sig]
        ax.bar(range(n_class), vals, color=colors)
        ax.axhline(0.5, color="k", lw=0.5, linestyle="--")
        ax.set_ylim(0, 1)
        ax.set_ylabel(concepts[k], rotation=0, ha="right", va="center", fontsize=8)
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.tick_params(axis="y", labelsize=7)
    axes[-1].set_xticks(range(n_class))
    axes[-1].set_xticklabels(classes, rotation=45, ha="right")
    fig.suptitle("Per-concept TCAV scores across classes (blue=significant)")
    fig.tight_layout()
    fig.savefig(out_dir / "tcav_per_concept_bar.pdf")
    plt.close(fig)
    print(f"[tcav] wrote {out_dir / 'tcav_per_concept_bar.pdf'}")


if __name__ == "__main__":
    main()
