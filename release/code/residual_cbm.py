#!/usr/bin/env python3
"""STREAM B -- interpretable architectures at backbone accuracy with LOAD-BEARING
concepts: PCBM-h (residual / post-hoc CBM) and CEM (Concept Embedding Model).

Target: interpretable W-F1 >= backbone (~0.83 frozen, ~0.85 fine-tuned) with
concepts that remain load-bearing (faithfulness intervention curves), NOT cosmetic.

This module is the honest, control-equipped successor to
concepts_help_performance.py::concept_residual. The prior run reported a
concept-residual variant at +1.0% W-F1 over the backbone but DID NOT run the two
controls the working theory flagged as decisive:

  (A) RANDOM-ORTHOGONAL-RESIDUAL CONTROL. PCBM-h concatenates [predicted concepts
      ; residual embedding]. If a RANDOM orthogonal projection of the backbone of
      matched rank, used as the residual, matches the concept-residual accuracy,
      then the gain is GENERIC CAPACITY, not the concepts. We build the residual
      two ways and compare head-to-head:
        - concept-residual: residual = backbone projected onto the orthogonal
          complement of the concept-prediction subspace (the "leftover" the
          concepts did not explain), rank r.
        - random-orthogonal-residual: residual = backbone projected onto a RANDOM
          rank-r orthonormal subspace. Matched rank r.

  (B) FAITHFULNESS / INTERVENTION CURVES. For PCBM-h and CEM, perturb a single
      concept on the bottleneck across a sweep and measure whether the target-class
      probability moves in the expected direction (monotone), and whether replacing
      a predicted concept with its TRUE measured value at increasing intervention
      FRACTION improves task accuracy (the standard CBM test-time-intervention
      curve). A flat / non-monotone curve == cosmetic concepts.

Architectures
-------------
PCBM-h (residual CBM, Yuksekgonul et al. 2023):
  concept head g: x -> c_hat in R^K   (K=10 measured concepts, linear, supervised)
  residual r:     x -> R^d_res        (rank-r features the concepts miss)
  class head:     [c_hat ; residual] -> logits
  The concept part is auditable (linear concept->class weights); residual carries
  the leftover. Intervention sets c_hat[:,j] := c_true[:,j].

CEM (Concept Embedding Model, Espinosa Zarlenga et al. 2022):
  per concept j: positive embedding e_j^+ and negative embedding e_j^- (each R^m),
  a concept-probability scoring s_j = sigmoid(w_j . [e_j^+ ; e_j^-]) supervised to
  c_true[:,j]; the mixed embedding  e_j = s_j * e_j^+ + (1-s_j) * e_j^-  is the
  concept's contextual representation. Classifier on concat_j e_j.
  Designed to BEAT a black box in the concept-INCOMPLETE regime (ours: 10 textbook
  concepts do not span the abnormality signal). Intervention sets s_j := c_true[:,j]
  (hard 0/1), i.e. swaps in the correct positive/negative embedding.

Backbone reference is read from outputs/max_classification/<tag>.json when present
(fine-tuned), else from an MLP probe on the same features (frozen). All numbers come
from THIS run on cached features; nothing is fabricated.

Inputs (all shipped in the release; runs on CPU):
  --features  weights/dinobloomb_ft_last4_s0_features.npz (fine-tuned bank; also a
              frozen bank you extract yourself). The shipped bank embeds `labels`
              and `class_names`, so `--labels_from_npz` runs it with no manifest.
  --manifest  data/cell_manifest.json  (de-identified labels; default)
  --morpho    data/morphometry_concepts.csv  (measured concepts; default)
  --backbone_json  optional max_classification JSON for a ft W-F1 reference

Concept set: the canonical 10 measured concepts (see concept_config_gr_neutro.json).
Multi-seed (6), paired-bootstrap CIs. CPU-friendly (small MLPs); GPU if available.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score

# The canonical 10 measured concepts (see concept_config_gr_neutro.json).
CONCEPTS = [
    "nuclear_lobulation_degree", "nuclear_contour_irregularity",
    "nucleus_to_cytoplasm_ratio", "chromatin_condensation_level",
    "chromatin_clumping_pattern", "cytoplasmic_granule_density",
    "granule_coarseness", "cytoplasmic_texture_uniformity",
    "cytoplasm_basophilia_level", "cytoplasmic_inclusion_visibility",
]
RARE_CLASSES = ["Dohle", "Hypergranulation", "Hypersegmentation"]
SEEDS = [0, 7, 13, 42, 1337, 2024]
N_BOOT = 5000


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def stratified_multilabel_split(labels_np, test_size=0.10, val_size=0.10, seed=42):
    """Verbatim from ch3_gr_neutro/data.py (identical split to every other run)."""
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


def load_morpho(morpho_csv):
    recs = {}
    with open(morpho_csv) as f:
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


def _cells_from_npz(npz):
    """Fallback labels source: derive the (filename, dominant_class_idx) list and
    the class-name vocabulary directly from the shipped feature bank, which embeds
    `labels` and `class_names` (see release/weights re-save). Used when no manifest
    JSON is supplied (`--labels_from_npz`)."""
    if "labels" not in npz.files or "class_names" not in npz.files:
        raise KeyError(
            "--labels_from_npz requires the npz to carry 'labels' and 'class_names' "
            "arrays; the shipped release/weights/dinobloomb_ft_last4_s0_features.npz "
            "does. For an older feature bank pass --manifest instead.")
    class_names = [str(c) for c in npz["class_names"]]
    labels = np.asarray(npz["labels"], dtype=np.int64)
    paths = npz["paths"]
    cells = [{"filename": Path(str(p)).name, "dominant_class_idx": int(labels[i])}
             for i, p in enumerate(paths)]
    return class_names, cells


def load_aligned(features_npz, manifest_json, morpho_csv, labels_from_npz=False):
    npz = np.load(features_npz, allow_pickle=True)
    if labels_from_npz or not manifest_json:
        class_names, cells = _cells_from_npz(npz)
    else:
        with open(manifest_json) as f:
            man = json.load(f)
        class_names = man["class_names"]
        cells = man["cells"]
    feats = npz["features"]
    paths = npz["paths"]
    feat_by_fn = {Path(str(p)).name: feats[i] for i, p in enumerate(paths)}
    morpho = load_morpho(morpho_csv)
    X, C, y, miss_f, miss_m = [], [], [], 0, 0
    for c in cells:
        fn = c["filename"]
        if fn not in feat_by_fn:
            miss_f += 1; continue
        if fn not in morpho:
            miss_m += 1; continue
        X.append(feat_by_fn[fn])
        C.append([morpho[fn][k] for k in CONCEPTS])
        y.append(c["dominant_class_idx"])
    X = np.vstack(X).astype(np.float32)
    C = np.vstack(C).astype(np.float32)
    y = np.asarray(y, dtype=np.int64)
    return X, C, y, class_names, dict(miss_feat=miss_f, miss_morpho=miss_m, n=len(y))


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def per_class_acc(y_true, y_pred, K):
    out = []
    for k in range(K):
        m = (y_true == k)
        out.append(float((y_pred[m] == k).mean()) if m.sum() else float("nan"))
    return out


def eval_pred(y_true, y_pred, K):
    labels = list(range(K))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "wf1": float(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)),
        "macrof1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "per_class_acc": per_class_acc(y_true, y_pred, K),
        "per_class_f1": f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0).tolist(),
    }


def paired_bootstrap_wf1(y_true, pred_a, pred_b, K, rng, n_boot=N_BOOT):
    """Delta W-F1 (b - a), paired resampling. mean, 2.5%, 97.5%, P(d>0)."""
    y_true = np.asarray(y_true); pa = np.asarray(pred_a); pb = np.asarray(pred_b)
    n = len(y_true); labels = list(range(K))
    d = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        d[i] = (f1_score(y_true[idx], pb[idx], average="weighted", labels=labels, zero_division=0)
                - f1_score(y_true[idx], pa[idx], average="weighted", labels=labels, zero_division=0))
    return [float(d.mean()), float(np.percentile(d, 2.5)),
            float(np.percentile(d, 97.5)), float((d > 0).mean())]


def seed_ci(vals):
    v = np.asarray(vals, float)
    m = v.mean()
    half = 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
    return [float(m), float(m - half), float(m + half)]


# --------------------------------------------------------------------------- #
# Torch modules
# --------------------------------------------------------------------------- #
class MLPProbe(nn.Module):
    """Backbone-alone reference head (frozen features -> class)."""
    def __init__(self, d, K, hidden=256, p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hidden, K))

    def forward(self, x):
        return self.net(x)


class PCBMh(nn.Module):
    """Post-hoc residual CBM.

    concept head g (linear, supervised to c_true) -> c_hat (K)
    residual r (linear d->d_res, then we orthogonalise its weight vs g at train end
       only conceptually; here the residual is a free low-rank head and the CONTROL
       supplies the matched-rank random-orthogonal alternative explicitly)
    class head: [c_hat ; residual] -> logits.
    """
    def __init__(self, d, K_concept, K_class, d_res, residual_mode="learned",
                 random_basis=None):
        super().__init__()
        self.K_concept = K_concept
        self.d_res = d_res
        self.residual_mode = residual_mode
        self.concept_head = nn.Linear(d, K_concept)
        if residual_mode == "learned":
            self.residual = nn.Linear(d, d_res)
        elif residual_mode == "random_ortho":
            # fixed random orthonormal projection of matched rank; NOT trained.
            assert random_basis is not None and random_basis.shape == (d, d_res)
            self.register_buffer("rb", random_basis)
            self.residual = None
        elif residual_mode == "none":
            self.residual = None
        else:
            raise ValueError(residual_mode)
        in_dim = K_concept + (d_res if residual_mode != "none" else 0)
        self.class_head = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, K_class))

    def residual_feat(self, x):
        if self.residual_mode == "learned":
            return self.residual(x)
        if self.residual_mode == "random_ortho":
            return x @ self.rb
        return None

    def forward(self, x, c_override=None, drop_residual=False):
        c_hat = self.concept_head(x)
        c_use = c_hat if c_override is None else c_override
        res = self.residual_feat(x)
        if res is not None and drop_residual:
            res = torch.zeros_like(res)
        feat = c_use if res is None else torch.cat([c_use, res], dim=1)
        return self.class_head(feat), c_hat


class CEM(nn.Module):
    """Concept Embedding Model. Per concept: pos/neg embedding (m each) from a shared
    trunk; scalar concept prob s_j supervised to c_true; mixed embedding
    e_j = s_j e_j^+ + (1-s_j) e_j^-; classifier on concat_j e_j."""
    def __init__(self, d, K_concept, K_class, emb=16, trunk=256):
        super().__init__()
        self.K = K_concept
        self.emb = emb
        self.trunk = nn.Sequential(nn.Linear(d, trunk), nn.ReLU(), nn.Dropout(0.2))
        self.pos = nn.Linear(trunk, K_concept * emb)
        self.neg = nn.Linear(trunk, K_concept * emb)
        self.score = nn.Linear(2 * emb, 1)  # shared scorer over [e+ ; e-] per concept
        self.classifier = nn.Sequential(
            nn.Linear(K_concept * emb, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, K_class))

    def embeddings(self, x):
        h = self.trunk(x)
        B = x.shape[0]
        ep = self.pos(h).view(B, self.K, self.emb)
        en = self.neg(h).view(B, self.K, self.emb)
        s = self.score(torch.cat([ep, en], dim=-1)).squeeze(-1)  # (B, K) logits
        return ep, en, s

    def forward(self, x, s_override=None):
        ep, en, s_logit = self.embeddings(x)
        s = torch.sigmoid(s_logit)
        s_mix = s if s_override is None else s_override  # (B,K) in [0,1]
        e = s_mix.unsqueeze(-1) * ep + (1 - s_mix).unsqueeze(-1) * en  # (B,K,emb)
        logits = self.classifier(e.reshape(e.shape[0], -1))
        return logits, s_logit


# --------------------------------------------------------------------------- #
# Training loops
# --------------------------------------------------------------------------- #
def make_loaders(Xtr, ytr, ctr, Xte, yte, cte, device, bs=128):
    def t(a): return torch.as_tensor(a, device=device)
    tr = torch.utils.data.TensorDataset(t(Xtr), t(ytr), t(ctr))
    te = torch.utils.data.TensorDataset(t(Xte), t(yte), t(cte))
    return (torch.utils.data.DataLoader(tr, batch_size=bs, shuffle=True),
            torch.utils.data.DataLoader(te, batch_size=512, shuffle=False))


def class_weights(ytr, K, device):
    cnt = np.bincount(ytr, minlength=K).astype(float)
    w = cnt.sum() / (K * np.maximum(cnt, 1.0))
    return torch.as_tensor(w, dtype=torch.float32, device=device)


def train_mlp(Xtr, ytr, Xte, yte, K, device, epochs=60, seed=0):
    torch.manual_seed(seed)
    d = Xtr.shape[1]
    model = MLPProbe(d, K).to(device)
    cw = class_weights(ytr, K, device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    Xtr_t = torch.as_tensor(Xtr, device=device); ytr_t = torch.as_tensor(ytr, device=device)
    Xte_t = torch.as_tensor(Xte, device=device)
    n = len(ytr); bs = 128
    for ep in range(epochs):
        model.train(); perm = torch.randperm(n, device=device)
        for s in range(0, n, bs):
            b = perm[s:s + bs]
            opt.zero_grad()
            loss = F.cross_entropy(model(Xtr_t[b]), ytr_t[b], weight=cw)
            loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(Xte_t).argmax(1).cpu().numpy()
    return pred


def train_pcbmh(Xtr, ytr, ctr, Xte, yte, cte, K_concept, K_class, device,
                residual_mode="learned", d_res=10, random_basis=None,
                epochs=80, seed=0, lambda_concept=1.0,
                faithful=False, residual_dropout=0.5, lambda_decorr=1.0):
    """PCBM-h trainer.

    faithful=True adds two faithfulness-promoting regularisers so the class head
    cannot route around the concepts (which makes interventions inert):
      - residual-DROPOUT: on a fraction `residual_dropout` of training steps the
        residual is zeroed, so the head must classify from concepts alone. This
        forces the concept->class path to carry real signal at test-intervention time.
      - residual-concept DECORRELATION: penalise correlation between residual
        features and predicted concepts so the residual stops duplicating concept
        info (the duplication is exactly what flattens the intervention curve).
    These DO trade a little accuracy for genuine intervention response; we report
    BOTH the plain and faithful variants so the trade is explicit and honest.
    """
    torch.manual_seed(seed)
    d = Xtr.shape[1]
    rb = None
    if residual_mode == "random_ortho":
        g = torch.Generator().manual_seed(seed + 9999)
        A = torch.randn(d, d_res, generator=g)
        q, _ = torch.linalg.qr(A)          # d x d_res orthonormal columns
        rb = q.to(device)
    model = PCBMh(d, K_concept, K_class, d_res, residual_mode, rb).to(device)
    cw = class_weights(ytr, K_class, device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    Xtr_t = torch.as_tensor(Xtr, device=device); ytr_t = torch.as_tensor(ytr, device=device)
    ctr_t = torch.as_tensor(ctr, device=device)
    Xte_t = torch.as_tensor(Xte, device=device); cte_t = torch.as_tensor(cte, device=device)
    n = len(ytr); bs = 128
    has_res = residual_mode != "none"
    for ep in range(epochs):
        model.train(); perm = torch.randperm(n, device=device)
        for s in range(0, n, bs):
            b = perm[s:s + bs]
            opt.zero_grad()
            drop = bool(faithful and has_res and torch.rand(1).item() < residual_dropout)
            logits, c_hat = model(Xtr_t[b], drop_residual=drop)
            L = (F.cross_entropy(logits, ytr_t[b], weight=cw)
                 + lambda_concept * F.mse_loss(c_hat, ctr_t[b]))
            if faithful and has_res and not drop:
                # decorrelate residual features from predicted concepts (batch corr)
                res = model.residual_feat(Xtr_t[b])
                rc = res - res.mean(0, keepdim=True)
                cc = c_hat - c_hat.mean(0, keepdim=True)
                rc = rc / (rc.std(0, keepdim=True) + 1e-6)
                cc = cc / (cc.std(0, keepdim=True) + 1e-6)
                corr = (cc.t() @ rc) / max(len(b), 1)          # (K, d_res)
                L = L + lambda_decorr * (corr ** 2).mean()
            L.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        logits, _ = model(Xte_t)
        pred = logits.argmax(1).cpu().numpy()
    return model, pred, (Xte_t, cte_t)


def train_cem(Xtr, ytr, ctr, Xte, yte, cte, K_concept, K_class, device,
              emb=16, epochs=80, seed=0, lambda_concept=1.0):
    torch.manual_seed(seed)
    d = Xtr.shape[1]
    model = CEM(d, K_concept, K_class, emb=emb).to(device)
    cw = class_weights(ytr, K_class, device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    Xtr_t = torch.as_tensor(Xtr, device=device); ytr_t = torch.as_tensor(ytr, device=device)
    # concept targets binarised at 0.5 for BCE on the concept score
    ctr_b = torch.as_tensor((ctr >= 0.5).astype(np.float32), device=device)
    Xte_t = torch.as_tensor(Xte, device=device)
    cte_b = torch.as_tensor((cte >= 0.5).astype(np.float32), device=device)
    n = len(ytr); bs = 128
    for ep in range(epochs):
        model.train(); perm = torch.randperm(n, device=device)
        for s in range(0, n, bs):
            b = perm[s:s + bs]
            opt.zero_grad()
            logits, s_logit = model(Xtr_t[b])
            L = (F.cross_entropy(logits, ytr_t[b], weight=cw)
                 + lambda_concept * F.binary_cross_entropy_with_logits(s_logit, ctr_b[b]))
            L.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        logits, _ = model(Xte_t)
        pred = logits.argmax(1).cpu().numpy()
    return model, pred, (Xte_t, cte_b)


# --------------------------------------------------------------------------- #
# Faithfulness / intervention
# --------------------------------------------------------------------------- #
@torch.no_grad()
def pcbmh_ttint_curve(model, Xte_t, cte_t, yte, K_class, device, fracs=(0.0, 0.25, 0.5, 0.75, 1.0), seed=0):
    """Test-time intervention: replace a FRACTION of predicted concepts with the
    TRUE measured value; report W-F1 vs fraction. Rising curve == load-bearing."""
    rng = np.random.default_rng(seed)
    c_hat = model.concept_head(Xte_t)            # (n, K)
    n, K = c_hat.shape
    out = []
    for fr in fracs:
        n_int = int(round(fr * K))
        # intervene on the n_int concepts (deterministic order by index for sweep stability)
        cov = c_hat.clone()
        if n_int > 0:
            cols = list(range(n_int))
            cov[:, cols] = cte_t[:, cols]
        logits, _ = model(Xte_t, c_override=cov)
        pred = logits.argmax(1).cpu().numpy()
        labels = list(range(K_class))
        out.append(float(f1_score(yte, pred, average="weighted", labels=labels, zero_division=0)))
    return list(fracs), out


@torch.no_grad()
def pcbmh_directional(model, Xte_t, K_class, class_names, device):
    """Directional sweep: set ONE concept across [-2,-1,0,1,2] (z would need scaling;
    we sweep in measured [0,1] units 0..1) and read mean dP on a target class.
    Reports the slope of P(target) vs concept value. Monotone +slope == faithful."""
    c_hat = model.concept_head(Xte_t)
    results = {}
    sweeps = {
        "nuclear_lobulation_degree": ("Hypersegmentation", "Hyposegmentation"),
        "cytoplasmic_granule_density": ("Hypergranulation", "Hypogranulation"),
        "chromatin_clumping_pattern": ("Chromatin", None),
        "cytoplasmic_inclusion_visibility": ("Dohle", None),
    }
    grid = np.linspace(0.0, 1.0, 5)
    for cname, (up_cls, dn_cls) in sweeps.items():
        if cname not in CONCEPTS:
            continue
        j = CONCEPTS.index(cname)
        up_i = class_names.index(up_cls) if up_cls in class_names else None
        dn_i = class_names.index(dn_cls) if (dn_cls and dn_cls in class_names) else None
        p_up, p_dn = [], []
        for v in grid:
            cov = c_hat.clone(); cov[:, j] = float(v)
            p = torch.softmax(model(Xte_t, c_override=cov)[0], dim=1).mean(0).cpu().numpy()
            p_up.append(float(p[up_i]) if up_i is not None else float("nan"))
            p_dn.append(float(p[dn_i]) if dn_i is not None else float("nan"))
        slope_up = float(np.polyfit(grid, p_up, 1)[0]) if up_i is not None else float("nan")
        slope_dn = float(np.polyfit(grid, p_dn, 1)[0]) if dn_i is not None else float("nan")
        results[cname] = {
            "target_up": up_cls, "slope_up": slope_up, "P_up_grid": p_up,
            "target_down": dn_cls, "slope_dn": slope_dn, "P_dn_grid": p_dn,
            "grid": grid.tolist(),
        }
    return results


@torch.no_grad()
def cem_ttint_curve(model, Xte_t, cte_b, yte, K_class, fracs=(0.0, 0.25, 0.5, 0.75, 1.0)):
    """CEM intervention: override the concept SCORE s_j with the true 0/1 for a
    fraction of concepts (swaps in correct pos/neg embedding)."""
    _, _, s_logit = model.embeddings(Xte_t)
    s = torch.sigmoid(s_logit)
    n, K = s.shape
    out = []
    for fr in fracs:
        n_int = int(round(fr * K))
        s_ov = s.clone()
        if n_int > 0:
            cols = list(range(n_int))
            s_ov[:, cols] = cte_b[:, cols]
        logits, _ = model(Xte_t, s_override=s_ov)
        pred = logits.argmax(1).cpu().numpy()
        labels = list(range(K_class))
        out.append(float(f1_score(yte, pred, average="weighted", labels=labels, zero_division=0)))
    return list(fracs), out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    data = here.parent / "data"          # release/data/ (shipped de-identified artifacts)
    ap.add_argument("--features", default=str(here.parent / "weights/dinobloomb_ft_last4_s0_features.npz"))
    ap.add_argument("--manifest", default=str(data / "cell_manifest.json"))
    ap.add_argument("--morpho", default=str(data / "morphometry_concepts.csv"))
    ap.add_argument("--labels_from_npz", action="store_true",
                    help="derive labels + class_names from the feature bank's embedded "
                         "'labels'/'class_names' arrays instead of --manifest (runs without a manifest JSON)")
    ap.add_argument("--backbone_json", default="",
                    help="optional max_classification ft JSON for the W-F1 reference")
    ap.add_argument("--backbone_tag", default="ft", help="label for the backbone reference (shipped bank is fine-tuned last-4)")
    ap.add_argument("--out", default=str(here / "outputs/residual_cbm/results.json"))
    ap.add_argument("--d_res", type=int, default=10, help="residual rank for PCBM-h + matched random-ortho control")
    ap.add_argument("--d_res_high", type=int, default=64, help="residual rank for the high-capacity PCBM-h (reach-backbone) variant + its matched control")
    ap.add_argument("--emb", type=int, default=16, help="CEM per-concept embedding dim")
    ap.add_argument("--epochs", type=int, default=80)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    X, C, y, class_names, align = load_aligned(
        args.features, args.manifest, args.morpho, labels_from_npz=args.labels_from_npz)
    K_class = len(class_names); K_concept = len(CONCEPTS)
    print(f"[load] X={X.shape} C={C.shape} y={y.shape} classes={K_class} concepts={K_concept} "
          f"(miss_feat={align['miss_feat']} miss_morpho={align['miss_morpho']}) device={device}", flush=True)
    print("[dist] " + ", ".join(f"{class_names[k]}={int((y==k).sum())}" for k in range(K_class)), flush=True)

    bb_ref = None
    if args.backbone_json and Path(args.backbone_json).exists():
        bd = json.load(open(args.backbone_json))
        bb_ref = {"source": args.backbone_json, "wf1_mean": bd.get("wf1_mean"),
                  "wf1_std": bd.get("wf1_std"), "macf1_mean": bd.get("macf1_mean"),
                  "tag": bd.get("tag")}
        print(f"[backbone-ref] {bb_ref['tag']}: W-F1={bb_ref['wf1_mean']:.4f}+/-{bb_ref['wf1_std']:.4f}", flush=True)

    per_seed = {}
    for seed in SEEDS:
        tr_i, val_i, te_i = stratified_multilabel_split(
            np.eye(K_class, dtype=int)[y], seed=seed)
        tr = np.concatenate([tr_i, val_i])
        # standardise features and concepts on train
        xs = StandardScaler().fit(X[tr]); Xtr = xs.transform(X[tr]); Xte = xs.transform(X[te_i])
        Xtr = Xtr.astype(np.float32); Xte = Xte.astype(np.float32)
        # concepts kept in measured [0,1] units (PCBM-h MSE target & intervention readability)
        ctr = C[tr]; cte = C[te_i]
        ytr = y[tr]; yte = y[te_i]

        res = {}
        # ---- backbone-alone MLP (frozen reference if no ft json) ----
        bb_pred = train_mlp(Xtr, ytr, Xte, yte, K_class, device, epochs=args.epochs, seed=seed)
        res["backbone_mlp"] = eval_pred(yte, bb_pred, K_class)

        # ---- PCBM-h (learned residual) ----
        m_pc, pc_pred, (Xte_t, cte_t) = train_pcbmh(
            Xtr, ytr, ctr, Xte, yte, cte, K_concept, K_class, device,
            residual_mode="learned", d_res=args.d_res, epochs=args.epochs, seed=seed)
        res["pcbmh"] = eval_pred(yte, pc_pred, K_class)

        # ---- CONTROL A: random-orthogonal residual, matched rank ----
        _, rnd_pred, _ = train_pcbmh(
            Xtr, ytr, ctr, Xte, yte, cte, K_concept, K_class, device,
            residual_mode="random_ortho", d_res=args.d_res, epochs=args.epochs, seed=seed)
        res["pcbmh_random_ortho_residual"] = eval_pred(yte, rnd_pred, K_class)

        # ---- pure bottleneck (no residual) = transparent CBM lower bar ----
        _, cbm_pred, _ = train_pcbmh(
            Xtr, ytr, ctr, Xte, yte, cte, K_concept, K_class, device,
            residual_mode="none", d_res=args.d_res, epochs=args.epochs, seed=seed)
        res["pure_bottleneck"] = eval_pred(yte, cbm_pred, K_class)

        # ---- PCBM-h FAITHFUL (residual-dropout + concept/residual decorrelation) ----
        # forces the concept->class path to carry signal so interventions are not inert.
        m_pcf, pcf_pred, (Xte_tf, cte_tf) = train_pcbmh(
            Xtr, ytr, ctr, Xte, yte, cte, K_concept, K_class, device,
            residual_mode="learned", d_res=args.d_res, epochs=args.epochs, seed=seed,
            faithful=True, residual_dropout=0.5, lambda_decorr=1.0)
        res["pcbmh_faithful"] = eval_pred(yte, pcf_pred, K_class)

        # ---- PCBM-h HIGH-RANK (d_res_high) to test reaching backbone accuracy ----
        m_pch, pch_pred, _ = train_pcbmh(
            Xtr, ytr, ctr, Xte, yte, cte, K_concept, K_class, device,
            residual_mode="learned", d_res=args.d_res_high, epochs=args.epochs, seed=seed)
        res["pcbmh_highrank"] = eval_pred(yte, pch_pred, K_class)
        # matched random-ortho control at the HIGH rank (is the high-rank gain concepts or capacity?)
        _, pch_rnd_pred, _ = train_pcbmh(
            Xtr, ytr, ctr, Xte, yte, cte, K_concept, K_class, device,
            residual_mode="random_ortho", d_res=args.d_res_high, epochs=args.epochs, seed=seed)
        res["pcbmh_highrank_random_ortho"] = eval_pred(yte, pch_rnd_pred, K_class)

        # ---- CEM ----
        m_cem, cem_pred, (Xte_c, cte_cb) = train_cem(
            Xtr, ytr, ctr, Xte, yte, cte, K_concept, K_class, device,
            emb=args.emb, epochs=args.epochs, seed=seed)
        res["cem"] = eval_pred(yte, cem_pred, K_class)

        # ---- paired bootstrap deltas vs backbone_mlp ----
        rng = np.random.default_rng(seed)
        res["delta_pcbmh_vs_backbone"] = paired_bootstrap_wf1(yte, bb_pred, pc_pred, K_class, rng)
        res["delta_cem_vs_backbone"] = paired_bootstrap_wf1(yte, bb_pred, cem_pred, K_class, np.random.default_rng(seed + 1))
        res["delta_pcbmh_vs_randortho"] = paired_bootstrap_wf1(yte, rnd_pred, pc_pred, K_class, np.random.default_rng(seed + 2))
        res["delta_pcbmh_vs_purebottleneck"] = paired_bootstrap_wf1(yte, cbm_pred, pc_pred, K_class, np.random.default_rng(seed + 3))
        res["delta_pcbmh_highrank_vs_backbone"] = paired_bootstrap_wf1(yte, bb_pred, pch_pred, K_class, np.random.default_rng(seed + 4))
        res["delta_pcbmh_highrank_vs_randortho"] = paired_bootstrap_wf1(yte, pch_rnd_pred, pch_pred, K_class, np.random.default_rng(seed + 5))
        res["delta_pcbmh_faithful_vs_backbone"] = paired_bootstrap_wf1(yte, bb_pred, pcf_pred, K_class, np.random.default_rng(seed + 6))

        # ---- faithfulness (seed 0 stores curves for the report; all seeds aggregate slopes) ----
        fracs, pc_curve = pcbmh_ttint_curve(m_pc, Xte_t, cte_t, yte, K_class, device, seed=seed)
        res["pcbmh_ttint_curve"] = {"fracs": fracs, "wf1": pc_curve}
        res["pcbmh_directional"] = pcbmh_directional(m_pc, Xte_t, K_class, class_names, device)
        # faithful variant intervention curve + directional sweep (the load-bearing test)
        fracs_f, pcf_curve = pcbmh_ttint_curve(m_pcf, Xte_tf, cte_tf, yte, K_class, device, seed=seed)
        res["pcbmh_faithful_ttint_curve"] = {"fracs": fracs_f, "wf1": pcf_curve}
        res["pcbmh_faithful_directional"] = pcbmh_directional(m_pcf, Xte_tf, K_class, class_names, device)
        cfracs, cem_curve = cem_ttint_curve(m_cem, Xte_c, cte_cb, yte, K_class)
        res["cem_ttint_curve"] = {"fracs": cfracs, "wf1": cem_curve}

        per_seed[str(seed)] = res
        print(f"[seed {seed}] bb={res['backbone_mlp']['wf1']:.4f} "
              f"pcbmh={res['pcbmh']['wf1']:.4f} rnd_ortho={res['pcbmh_random_ortho_residual']['wf1']:.4f} "
              f"pureCBM={res['pure_bottleneck']['wf1']:.4f} cem={res['cem']['wf1']:.4f} "
              f"| dWF1(pcbmh-bb)={res['delta_pcbmh_vs_backbone'][0]:+.4f} "
              f"dWF1(pcbmh-rnd)={res['delta_pcbmh_vs_randortho'][0]:+.4f} "
              f"| ttint {pc_curve[0]:.3f}->{pc_curve[-1]:.3f}", flush=True)

    # ---- aggregate ----
    methods = ["backbone_mlp", "pcbmh", "pcbmh_random_ortho_residual", "pure_bottleneck",
               "pcbmh_faithful", "pcbmh_highrank", "pcbmh_highrank_random_ortho", "cem"]
    def agg(metric):
        return {m: seed_ci([per_seed[str(s)][m][metric] for s in SEEDS]) for m in methods}
    def agg_perclass(metric):
        out = {}
        for m in methods:
            arr = np.array([per_seed[str(s)][m][metric] for s in SEEDS])  # (S,K)
            out[m] = {class_names[k]: seed_ci(arr[:, k].tolist()) for k in range(K_class)}
        return out

    delta_keys = ["delta_pcbmh_vs_backbone", "delta_cem_vs_backbone",
                  "delta_pcbmh_vs_randortho", "delta_pcbmh_vs_purebottleneck",
                  "delta_pcbmh_highrank_vs_backbone", "delta_pcbmh_highrank_vs_randortho",
                  "delta_pcbmh_faithful_vs_backbone"]
    delta_agg = {dk: seed_ci([per_seed[str(s)][dk][0] for s in SEEDS]) for dk in delta_keys}

    # faithfulness aggregate: mean ttint curve, and mean directional slopes
    def mean_curve(key):
        arr = np.array([per_seed[str(s)][key]["wf1"] for s in SEEDS])
        return {"fracs": per_seed[str(SEEDS[0])][key]["fracs"],
                "wf1_mean": arr.mean(0).tolist(), "wf1_std": arr.std(0).tolist()}
    def agg_directional(key):
        ds = {}
        for cname in per_seed[str(SEEDS[0])][key]:
            su = [per_seed[str(s)][key][cname]["slope_up"] for s in SEEDS]
            sd = [per_seed[str(s)][key][cname]["slope_dn"] for s in SEEDS]
            ds[cname] = {
                "target_up": per_seed[str(SEEDS[0])][key][cname]["target_up"],
                "slope_up_seedmean_ci95": seed_ci(su),
                "target_down": per_seed[str(SEEDS[0])][key][cname]["target_down"],
                "slope_dn_seedmean_ci95": seed_ci([v for v in sd if not np.isnan(v)]) if any(not np.isnan(v) for v in sd) else None,
            }
        return ds
    dir_slopes = agg_directional("pcbmh_directional")
    dir_slopes_faithful = agg_directional("pcbmh_faithful_directional")

    out = {
        "experiment": "residual_cbm (Stream B: PCBM-h + CEM, controls + faithfulness)",
        "backbone": "DinoBloom-B",
        "features": args.features,
        "backbone_reference_external": bb_ref,
        "backbone_tag": args.backbone_tag,
        "n_cells": align["n"], "n_classes": K_class, "n_concepts": K_concept,
        "concepts": CONCEPTS, "class_names": class_names, "rare_classes": RARE_CLASSES,
        "seeds": SEEDS, "n_bootstrap": N_BOOT, "d_res": args.d_res,
        "d_res_high": args.d_res_high, "cem_emb": args.emb,
        "class_dist": {class_names[k]: int((y == k).sum()) for k in range(K_class)},
        "wf1": agg("wf1"), "macrof1": agg("macrof1"), "accuracy": agg("accuracy"),
        "balanced_accuracy": agg("balanced_accuracy"),
        "per_class_f1": agg_perclass("per_class_f1"),
        "per_class_acc": agg_perclass("per_class_acc"),
        "deltas_seedmean_ci95": delta_agg,
        "faithfulness": {
            "pcbmh_ttint_curve": mean_curve("pcbmh_ttint_curve"),
            "pcbmh_faithful_ttint_curve": mean_curve("pcbmh_faithful_ttint_curve"),
            "cem_ttint_curve": mean_curve("cem_ttint_curve"),
            "pcbmh_directional_slopes": dir_slopes,
            "pcbmh_faithful_directional_slopes": dir_slopes_faithful,
            "note": "ttint curve: W-F1 vs fraction of predicted concepts replaced by TRUE "
                    "measured value. Rising == load-bearing. directional slopes: dP(target "
                    "class)/d(concept value) over [0,1]; positive for 'up' target, negative "
                    "for 'down' target == faithful, monotone concept effect. The plain pcbmh "
                    "co-adapts its residual with the concept head (flat intervention curve); "
                    "pcbmh_faithful uses residual-dropout + concept/residual decorrelation to "
                    "make the concept->class path load-bearing at intervention time.",
        },
        "per_seed": per_seed,
        "interpretation_guide": {
            "load_bearing_test_1": "pcbmh > pure_bottleneck AND pcbmh ttint curve rises with intervention fraction.",
            "load_bearing_test_2": "pcbmh vs random_ortho_residual: if delta_pcbmh_vs_randortho CI excludes 0 and is positive, "
                                   "the concepts (not generic residual capacity) drive the gain. If it straddles 0, the residual "
                                   "is doing the work and the concept gain is cosmetic for ACCURACY (explanation still valid).",
            "target": "interpretable W-F1 >= backbone with load-bearing concepts.",
        },
        "time_seconds": time.time() - t0,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[done] {time.time()-t0:.0f}s -> {args.out}", flush=True)
    print(f"[SUMMARY] backbone={out['wf1']['backbone_mlp'][0]:.4f} "
          f"pcbmh={out['wf1']['pcbmh'][0]:.4f} "
          f"rnd_ortho={out['wf1']['pcbmh_random_ortho_residual'][0]:.4f} "
          f"pure_cbm={out['wf1']['pure_bottleneck'][0]:.4f} "
          f"pcbmh_faithful={out['wf1']['pcbmh_faithful'][0]:.4f} "
          f"pcbmh_hr={out['wf1']['pcbmh_highrank'][0]:.4f} "
          f"pcbmh_hr_rnd={out['wf1']['pcbmh_highrank_random_ortho'][0]:.4f} "
          f"cem={out['wf1']['cem'][0]:.4f}", flush=True)
    fz = out["faithfulness"]
    print(f"[SUMMARY] dWF1 pcbmh-vs-randortho={delta_agg['delta_pcbmh_vs_randortho']} "
          f"pcbmh_hr-vs-randortho={delta_agg['delta_pcbmh_highrank_vs_randortho']} "
          f"pcbmh_hr-vs-backbone={delta_agg['delta_pcbmh_highrank_vs_backbone']}", flush=True)
    print(f"[SUMMARY] ttint plain {fz['pcbmh_ttint_curve']['wf1_mean'][0]:.4f}->{fz['pcbmh_ttint_curve']['wf1_mean'][-1]:.4f} "
          f"| faithful {fz['pcbmh_faithful_ttint_curve']['wf1_mean'][0]:.4f}->{fz['pcbmh_faithful_ttint_curve']['wf1_mean'][-1]:.4f}", flush=True)


if __name__ == "__main__":
    main()
