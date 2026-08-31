#!/usr/bin/env python3
"""Does FUSING / CO-TRAINING measured morphology concepts with the DinoBloom-B
backbone make GR-Neutro 7-class classification MORE accurate than the backbone
alone, while keeping the prediction explainable?

Central thesis under test: explainability should HELP performance, not cost it.

We report PLAIN ACCURACY (% correct) -- overall + per-class + confusion matrix --
because the target audience (biologists) does not read AUC/F1. (W-F1 / macro-F1
are also logged for continuity with prior runs, but accuracy is the headline.)

Inputs (all cached on Ruche, SLURM-only -- never the login node):
  - outputs/dinobloom_features.npz          : (4378, 768) frozen DinoBloom-B CLS features
  - outputs/morphometry_concepts/...csv     : 11 DIRECTLY-MEASURED morphology concepts/cell
  - outputs/p3_pilot/cell_manifest_full_extended.json : labels (dominant_class_idx, 7 classes)

Four concept-as-PERFORMANCE-ASSET variants, all vs backbone-alone, 6 seeds, paired
bootstrap CIs:

  V1  FEATURE FUSION          : concat dino(768) (+) measured concepts -> LogReg / HistGBM.
                                Black-box-ish (concepts are extra inputs) but a clean test
                                of whether the concept signal raises accuracy at all.
  V2  MULTI-TASK / AUXILIARY  : shared MLP trunk on dino features -> {class head, concept
                                regression head}. Aux concept loss with tunable weight.
                                EXPLAINABLE: the model predicts the concepts as a side output.
                                Does the shared representation classify better, esp. on the
                                rare morphology-defined classes?
  V3  CONCEPT-RESIDUAL (CEM)  : concept-bottleneck (predict concepts, classify THROUGH them)
                                PLUS a learned residual channel. Concepts give the explanation;
                                residual preserves accuracy. The literature's no-accuracy-drop
                                recipe. We also log the residual=0 (pure-CBM, fully transparent)
                                accuracy so the explainability<->accuracy trade is explicit.
  V4  RARE-CLASS FOCUS        : per-class accuracy delta (variant - backbone) for every variant,
                                with emphasis on Dohle / Hypersegmentation / Hypergranulation
                                where morphology is the key signal and the backbone is weakest.

NO fabrication. Honest: if no variant beats the backbone, that is reported plainly.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (f1_score, accuracy_score, balanced_accuracy_score,
                             confusion_matrix)

HERE = Path(__file__).resolve().parent
OUT_ROOT = HERE / "outputs"
FEATS_NPZ = OUT_ROOT / "dinobloom_features.npz"
MANIFEST = OUT_ROOT / "p3_pilot" / "cell_manifest_full_extended.json"
MORPHO_CSV = OUT_ROOT / "morphometry_concepts" / "morphometry_concepts.csv"
OUT_DIR = OUT_ROOT / "concepts_help_performance"

# 11 directly-MEASURED concept columns (the computed concept VALUES in [0,1]).
CONCEPTS = [
    "nuclear_lobulation_degree", "nuclear_contour_irregularity",
    "nucleus_to_cytoplasm_ratio", "chromatin_condensation_level",
    "chromatin_clumping_pattern", "cytoplasmic_granule_density",
    "granule_coarseness", "cytoplasmic_texture_uniformity",
    "cytoplasm_basophilia_level", "cytoplasmic_inclusion_visibility",
    "cytoplasmic_vacuolization_degree",
]
# Reliable subset (concept->class AUC>=0.60 & non-degenerate variation), per the
# morphometry validation. Used for a "reliable-concepts-only" fusion variant.
RELIABLE = [
    "nuclear_lobulation_degree", "nuclear_contour_irregularity",
    "chromatin_clumping_pattern", "cytoplasmic_granule_density",
    "cytoplasmic_texture_uniformity", "cytoplasm_basophilia_level",
    "cytoplasmic_vacuolization_degree",
]

SEEDS = [0, 7, 13, 42, 1337, 2024]   # >=5 seeds
N_BOOT = 5000
RARE_CLASSES = ["Dohle", "Hypergranulation", "Hypersegmentation"]


# --------------------------------------------------------------------------- #
# Split: verbatim from ch3_gr_neutro/data.py (same as faithful_concept_classifier).
# --------------------------------------------------------------------------- #
def stratified_multilabel_split(labels_np, test_size=0.10, val_size=0.10, seed=42):
    from sklearn.model_selection import StratifiedShuffleSplit
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


def load_morpho_by_filename():
    recs = {}
    with open(MORPHO_CSV) as f:
        for row in csv.DictReader(f):
            if str(row.get("seg_ok", "1")).strip() not in ("1", "1.0", "True", "true"):
                continue
            d, ok = {}, True
            for c in CONCEPTS:
                try:
                    d[c] = float(row.get(c, ""))
                except (TypeError, ValueError):
                    ok = False
                    break
            if ok:
                recs[row["filename"]] = d
    return recs


# --------------------------------------------------------------------------- #
# Metrics helpers -- PLAIN ACCURACY is the headline.
# --------------------------------------------------------------------------- #
def per_class_accuracy(y_true, y_pred, n_classes):
    """Recall per class = fraction of class-k cells predicted correctly = the
    per-class diagonal of a row-normalised confusion matrix. This is what a
    biologist means by 'accuracy on Dohle cells'."""
    out = []
    for k in range(n_classes):
        mask = (y_true == k)
        out.append(float((y_pred[mask] == k).mean()) if mask.sum() else float("nan"))
    return out


def eval_pred(y_true, y_pred, n_classes):
    labels = list(range(n_classes))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "wf1": float(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)),
        "macrof1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "per_class_acc": per_class_accuracy(y_true, y_pred, n_classes),
        "per_class_f1": f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0).tolist(),
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def paired_bootstrap_acc(y_true, pred_base, pred_var, n_classes, rng, n_boot=N_BOOT):
    """Delta accuracy (variant - backbone), overall + per-class, with 95% CI and
    P(delta>0). Paired: same resampled test indices for both predictors.

    Fully vectorised (no per-resample sklearn calls): we precompute correctness
    masks and a one-hot of the true class, then aggregate over a single
    (n_boot, n) index matrix with bincount-style sums.
    """
    y_true = np.asarray(y_true); pa = np.asarray(pred_base); pb = np.asarray(pred_var)
    n = len(y_true)
    correct_a = (pa == y_true).astype(np.float64)       # (n,)
    correct_b = (pb == y_true).astype(np.float64)
    onehot = np.zeros((n, n_classes), dtype=np.float64)  # (n, K) true-class indicator
    onehot[np.arange(n), y_true] = 1.0

    # chunk over resamples to cap the (chunk, n, K) buffer at a few hundred MB
    chunk = max(1, min(n_boot, int(4e7 / max(n * n_classes, 1))))
    d_acc = np.empty(n_boot); d_bal = np.empty(n_boot)
    d_pc = np.empty((n_boot, n_classes))
    for s in range(0, n_boot, chunk):
        e = min(s + chunk, n_boot)
        idx = rng.integers(0, n, size=(e - s, n))        # (b, n)
        d_acc[s:e] = correct_b[idx].mean(axis=1) - correct_a[idx].mean(axis=1)
        oh = onehot[idx]                                 # (b, n, K)
        class_count = oh.sum(axis=1)                      # (b, K)
        ca = (oh * correct_a[idx][:, :, None]).sum(axis=1)
        cb = (oh * correct_b[idx][:, :, None]).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            pacc_a = ca / class_count
            pacc_b = cb / class_count
        d_pc[s:e] = pacc_b - pacc_a
        d_bal[s:e] = np.nanmean(pacc_b, axis=1) - np.nanmean(pacc_a, axis=1)

    def ci1(arr):
        arr = np.asarray(arr, dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            return [float("nan")] * 4
        return [float(arr.mean()), float(np.percentile(arr, 2.5)),
                float(np.percentile(arr, 97.5)), float((arr > 0).mean())]

    return {
        "d_accuracy": ci1(d_acc),
        "d_balanced_accuracy": ci1(d_bal),
        "d_per_class_acc": [ci1(d_pc[:, k]) for k in range(n_classes)],
    }


def seed_ci(vals):
    vals = np.asarray(vals, dtype=float)
    m = float(np.nanmean(vals))
    if np.sum(~np.isnan(vals)) > 1:
        half = 1.96 * np.nanstd(vals, ddof=1) / np.sqrt(np.sum(~np.isnan(vals)))
    else:
        half = 0.0
    return [m, m - half, m + half]


# --------------------------------------------------------------------------- #
# V1 -- feature fusion heads
# --------------------------------------------------------------------------- #
def fit_logreg(Xtr, ytr, Xte, seed):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced",
                             multi_class="multinomial", random_state=seed)
    clf.fit(sc.transform(Xtr), ytr)
    return clf.predict(sc.transform(Xte))


def fit_gbm(Xtr, ytr, Xte, seed):
    clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06,
                                         max_leaf_nodes=31, l2_regularization=1.0,
                                         class_weight="balanced", random_state=seed)
    clf.fit(Xtr, ytr)
    return clf.predict(Xte)


# --------------------------------------------------------------------------- #
# V2 -- multi-task: shared MLP trunk -> {class head, concept regression head}
# --------------------------------------------------------------------------- #
class MultiTaskNet(nn.Module):
    def __init__(self, d_in, n_classes, n_concepts, hidden=256, p=0.4):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(d_in, hidden), nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(p),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(p),
        )
        self.cls_head = nn.Linear(hidden, n_classes)
        self.concept_head = nn.Linear(hidden, n_concepts)  # regression onto measured concepts

    def forward(self, x):
        h = self.trunk(x)
        return self.cls_head(h), self.concept_head(h)


def train_multitask(Xtr, ytr, Ctr, Xte, n_classes, n_concepts, class_w, seed,
                    aux_weight=0.0, epochs=120, device="cpu"):
    """aux_weight=0.0 => single-task baseline MLP (no concept supervision).
    aux_weight>0 => co-train class + concept regression. Returns test predictions."""
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.long, device=device)
    Ctr_t = torch.tensor(Ctr, dtype=torch.float32, device=device)
    Xte_t = torch.tensor(Xte, dtype=torch.float32, device=device)
    cw = torch.tensor(class_w, dtype=torch.float32, device=device)
    net = MultiTaskNet(Xtr.shape[1], n_classes, n_concepts).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ce = nn.CrossEntropyLoss(weight=cw)
    n = len(Xtr_t); bs = 256
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            logits, chat = net(Xtr_t[b])
            loss = ce(logits, ytr_t[b])
            if aux_weight > 0:
                loss = loss + aux_weight * F.mse_loss(chat, Ctr_t[b])
            loss.backward(); opt.step()
        sched.step()
    net.eval()
    with torch.no_grad():
        logits, _ = net(Xte_t)
        return logits.argmax(1).cpu().numpy()


# --------------------------------------------------------------------------- #
# V3 -- concept-residual (CEM / post-hoc-CBM style)
#   concept-bottleneck:  x -> chat (predicted concepts)  -> class logits (W_c)
#   residual channel:     x -> residual emb              -> class logits (W_r)
#   final = W_c(chat) + W_r(residual). Setting residual=0 recovers a pure,
#   fully-transparent CBM whose accuracy we also report.
# --------------------------------------------------------------------------- #
class ConceptResidualNet(nn.Module):
    def __init__(self, d_in, n_classes, n_concepts, res_dim=64, hidden=256, p=0.4):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(d_in, hidden), nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(p),
        )
        # concept predictor (the interpretable bottleneck)
        self.concept_head = nn.Linear(hidden, n_concepts)
        # class-from-concepts (the explainable path)
        self.cls_from_concepts = nn.Linear(n_concepts, n_classes)
        # residual path (preserves accuracy; carries what concepts cannot)
        self.residual = nn.Sequential(nn.Linear(hidden, res_dim), nn.GELU())
        self.cls_from_residual = nn.Linear(res_dim, n_classes)

    def forward(self, x, use_residual=True):
        h = self.trunk(x)
        chat = self.concept_head(h)
        logits_c = self.cls_from_concepts(chat)
        if use_residual:
            r = self.residual(h)
            logits = logits_c + self.cls_from_residual(r)
        else:
            logits = logits_c
        return logits, chat, logits_c


def train_concept_residual(Xtr, ytr, Ctr, Xte, n_classes, n_concepts, class_w, seed,
                           concept_weight=1.0, epochs=120, device="cpu"):
    """Returns (pred_full, pred_cbm_only) -- with and without the residual channel."""
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.long, device=device)
    Ctr_t = torch.tensor(Ctr, dtype=torch.float32, device=device)
    Xte_t = torch.tensor(Xte, dtype=torch.float32, device=device)
    cw = torch.tensor(class_w, dtype=torch.float32, device=device)
    net = ConceptResidualNet(Xtr.shape[1], n_classes, n_concepts).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ce = nn.CrossEntropyLoss(weight=cw)
    n = len(Xtr_t); bs = 256
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            logits, chat, logits_c = net(Xtr_t[b], use_residual=True)
            # supervise: full classifier + concept regression + concept-only classifier
            # (so the bottleneck stays predictive on its own -> explainable)
            loss = (ce(logits, ytr_t[b])
                    + 0.5 * ce(logits_c, ytr_t[b])
                    + concept_weight * F.mse_loss(chat, Ctr_t[b]))
            loss.backward(); opt.step()
        sched.step()
    net.eval()
    with torch.no_grad():
        logits_full, _, logits_c = net(Xte_t, use_residual=True)
        pred_full = logits_full.argmax(1).cpu().numpy()
        pred_cbm = logits_c.argmax(1).cpu().numpy()
    return pred_full, pred_cbm


# --------------------------------------------------------------------------- #
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}", flush=True)

    man = json.load(open(MANIFEST))
    class_names = man["class_names"]
    n_classes = len(class_names)
    cells = man["cells"]

    npz = np.load(FEATS_NPZ, allow_pickle=True)
    feats, paths = npz["features"], npz["paths"]
    feat_by_fn = {Path(str(p)).name: feats[i] for i, p in enumerate(paths)}
    morpho = load_morpho_by_filename()
    print(f"[align] cells={len(cells)} dino={len(feat_by_fn)} morpho={len(morpho)}", flush=True)

    rel_idx = [CONCEPTS.index(c) for c in RELIABLE]
    keep, X_dino, M_all, y, labels_oh = [], [], [], [], []
    miss_f = miss_m = 0
    for c in cells:
        fn = c["filename"]
        if fn not in feat_by_fn:
            miss_f += 1; continue
        if fn not in morpho:
            miss_m += 1; continue
        keep.append(fn)
        X_dino.append(feat_by_fn[fn])
        M_all.append([morpho[fn][k] for k in CONCEPTS])
        y.append(c["dominant_class_idx"])
        labels_oh.append(c["label_one_hot"])
    X_dino = np.vstack(X_dino).astype(np.float32)
    M_all = np.vstack(M_all).astype(np.float32)
    M_rel = M_all[:, rel_idx]
    y = np.asarray(y); labels_oh = np.asarray(labels_oh)
    n_concepts = M_all.shape[1]
    print(f"[align] kept={len(keep)} miss_feat={miss_f} miss_morpho={miss_m}", flush=True)
    print("[align] class dist: " + ", ".join(
        f"{class_names[k]}={int((y==k).sum())}" for k in range(n_classes)), flush=True)

    # variants that produce a single prediction vector per seed
    per_seed = {}
    for seed in SEEDS:
        tr_idx, val_idx, te_idx = stratified_multilabel_split(labels_oh, seed=seed)
        tr = np.concatenate([tr_idx, val_idx])
        yte = y[te_idx]

        # class weights for the NN heads (inverse frequency on train)
        counts = np.bincount(y[tr], minlength=n_classes).astype(float)
        class_w = (counts.sum() / (n_classes * np.maximum(counts, 1.0)))

        # standardise dino features once (used by NN variants)
        sc = StandardScaler().fit(X_dino[tr])
        Xtr_s = sc.transform(X_dino[tr]).astype(np.float32)
        Xte_s = sc.transform(X_dino[te_idx]).astype(np.float32)
        # standardise concept targets (regression target for V2/V3)
        csc = StandardScaler().fit(M_all[tr])
        Ctr_s = csc.transform(M_all[tr]).astype(np.float32)

        preds = {}
        # ---- BACKBONE-ALONE references ----
        preds["backbone_logreg"] = fit_logreg(X_dino[tr], y[tr], X_dino[te_idx], seed)
        preds["backbone_gbm"] = fit_gbm(X_dino[tr], y[tr], X_dino[te_idx], seed)
        preds["backbone_mlp"] = train_multitask(
            Xtr_s, y[tr], Ctr_s, Xte_s, n_classes, n_concepts, class_w, seed,
            aux_weight=0.0, device=device)

        # ---- V1 FEATURE FUSION ----
        Xcat = np.hstack([X_dino, M_all])
        Xcat_rel = np.hstack([X_dino, M_rel])
        preds["fusion_logreg"] = fit_logreg(Xcat[tr], y[tr], Xcat[te_idx], seed)
        preds["fusion_logreg_reliable"] = fit_logreg(Xcat_rel[tr], y[tr], Xcat_rel[te_idx], seed)
        preds["fusion_gbm"] = fit_gbm(Xcat[tr], y[tr], Xcat[te_idx], seed)

        # ---- V2 MULTI-TASK (sweep aux weight) ----
        for aw in (0.3, 1.0, 3.0):
            tag = f"multitask_aux{aw}".replace(".", "p")
            preds[tag] = train_multitask(
                Xtr_s, y[tr], Ctr_s, Xte_s, n_classes, n_concepts, class_w, seed,
                aux_weight=aw, device=device)

        # ---- V3 CONCEPT-RESIDUAL (CEM) ----
        pf, pc = train_concept_residual(
            Xtr_s, y[tr], Ctr_s, Xte_s, n_classes, n_concepts, class_w, seed,
            concept_weight=1.0, device=device)
        preds["concept_residual_full"] = pf      # concepts + residual (explainable + accurate)
        preds["concept_residual_cbmonly"] = pc   # residual OFF = pure transparent CBM

        # ---- evaluate everything ----
        seed_res = {name: eval_pred(yte, p, n_classes) for name, p in preds.items()}

        # ---- paired bootstrap deltas vs the strongest backbone reference ----
        # pick backbone reference = best backbone-alone head by THIS seed's accuracy
        bb_keys = ["backbone_logreg", "backbone_gbm", "backbone_mlp"]
        bb_ref = max(bb_keys, key=lambda k: seed_res[k]["accuracy"])
        rng = np.random.default_rng(seed)
        deltas = {}
        for name in preds:
            if name in bb_keys:
                continue
            deltas[name] = paired_bootstrap_acc(
                yte, preds[bb_ref], preds[name], n_classes, rng)
        seed_res["_bb_ref"] = bb_ref
        seed_res["_deltas_vs_bbref"] = deltas
        per_seed[str(seed)] = seed_res

        line = (f"[seed {seed}] bb_lr_acc={seed_res['backbone_logreg']['accuracy']:.4f} "
                f"bb_gbm={seed_res['backbone_gbm']['accuracy']:.4f} "
                f"bb_mlp={seed_res['backbone_mlp']['accuracy']:.4f} | "
                f"fusion_lr={seed_res['fusion_logreg']['accuracy']:.4f} "
                f"fusion_gbm={seed_res['fusion_gbm']['accuracy']:.4f} "
                f"mt_aux1={seed_res['multitask_aux1p0']['accuracy']:.4f} "
                f"cem_full={seed_res['concept_residual_full']['accuracy']:.4f} "
                f"cem_cbm={seed_res['concept_residual_cbmonly']['accuracy']:.4f}")
        print(line, flush=True)

    # ----------------------------- aggregate ----------------------------- #
    variant_names = [k for k in per_seed[str(SEEDS[0])] if not k.startswith("_")]

    def agg_scalar(name, metric):
        vals = [per_seed[str(s)][name][metric] for s in SEEDS]
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                "ci95": seed_ci(vals), "per_seed": [float(v) for v in vals]}

    def agg_per_class(name, metric):
        # metric in {"per_class_acc","per_class_f1"}
        arr = np.array([per_seed[str(s)][name][metric] for s in SEEDS], dtype=float)
        return {class_names[k]: {"mean": float(np.nanmean(arr[:, k])),
                                 "std": float(np.nanstd(arr[:, k]))}
                for k in range(n_classes)}

    def agg_confusion(name):
        arr = np.array([per_seed[str(s)][name]["confusion"] for s in SEEDS], dtype=float)
        return arr.sum(axis=0).astype(int).tolist()  # summed over seeds

    summary = {
        "n_cells": int(len(keep)), "n_classes": n_classes,
        "class_names": class_names, "concepts": CONCEPTS,
        "n_concepts": n_concepts, "seeds": SEEDS,
        "class_dist": {class_names[k]: int((y == k).sum()) for k in range(n_classes)},
        "rare_classes": RARE_CLASSES,
        "variants": {},
    }
    for name in variant_names:
        summary["variants"][name] = {
            "accuracy": agg_scalar(name, "accuracy"),
            "balanced_accuracy": agg_scalar(name, "balanced_accuracy"),
            "wf1": agg_scalar(name, "wf1"),
            "macrof1": agg_scalar(name, "macrof1"),
            "per_class_acc": agg_per_class(name, "per_class_acc"),
            "per_class_f1": agg_per_class(name, "per_class_f1"),
            "confusion_summed_over_seeds": agg_confusion(name),
        }

    # backbone reference for the report = best mean-accuracy backbone-alone head
    bb_keys = ["backbone_logreg", "backbone_gbm", "backbone_mlp"]
    bb_ref = max(bb_keys, key=lambda k: summary["variants"][k]["accuracy"]["mean"])
    summary["backbone_reference"] = bb_ref

    # delta-accuracy (seed-mean of per-seed paired-bootstrap means) for each variant
    delta_summary = {}
    for name in variant_names:
        if name in bb_keys:
            continue
        d_acc = [per_seed[str(s)]["_deltas_vs_bbref"][name]["d_accuracy"][0] for s in SEEDS]
        d_bal = [per_seed[str(s)]["_deltas_vs_bbref"][name]["d_balanced_accuracy"][0] for s in SEEDS]
        pgt0 = [per_seed[str(s)]["_deltas_vs_bbref"][name]["d_accuracy"][3] for s in SEEDS]
        per_class_d = {}
        for k in range(n_classes):
            vals = [per_seed[str(s)]["_deltas_vs_bbref"][name]["d_per_class_acc"][k][0] for s in SEEDS]
            per_class_d[class_names[k]] = seed_ci(vals)
        delta_summary[name] = {
            "d_accuracy_seedmean_ci95": seed_ci(d_acc),
            "d_balanced_accuracy_seedmean_ci95": seed_ci(d_bal),
            "mean_P_delta_gt0": float(np.mean(pgt0)),
            "per_class_d_accuracy_seedmean": per_class_d,
        }
    summary["deltas_vs_backbone"] = delta_summary
    summary["time_seconds"] = time.time() - t0

    (OUT_DIR / "results.json").write_text(json.dumps(
        {"summary": summary, "per_seed": per_seed}, indent=2))
    print(f"[save] {OUT_DIR}/results.json  ({time.time()-t0:.0f}s)", flush=True)

    # quick stdout headline
    bb_acc = summary["variants"][bb_ref]["accuracy"]["mean"]
    print(f"\n[HEADLINE] backbone-alone ({bb_ref}) accuracy = {bb_acc:.4f}", flush=True)
    for name in variant_names:
        if name in bb_keys:
            continue
        a = summary["variants"][name]["accuracy"]["mean"]
        d = delta_summary[name]["d_accuracy_seedmean_ci95"]
        print(f"  {name:28s} acc={a:.4f}  dAcc={d[0]:+.4f} "
              f"[{d[1]:+.4f},{d[2]:+.4f}] P(>0)={delta_summary[name]['mean_P_delta_gt0']:.2f}",
              flush=True)


if __name__ == "__main__":
    main()
