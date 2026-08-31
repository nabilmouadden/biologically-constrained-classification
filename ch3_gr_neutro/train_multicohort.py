"""Multi-cohort CBM training with cohort-ID auxiliary head (iVAE conditioning).

Operationalises Contribution C1 — multi-cohort training as the iVAE oracle.

Setup:
  Cohorts u in {0: GR-Neutro, 1: AML Matek, 2: MLL-23} carry per-cell DinoBloom-B
  CLS-token features cached on disk. GR-Neutro has full concept + class
  supervision (11 textbook concepts, 7 morphology classes). AML Matek has class
  supervision (15 cyto classes) AND a textbook-prior concept target derived
  from its class label via the GR-Neutro class->concept matrix (when its class
  has a mapping). MLL-23 has class supervision only (no concept labels).

Model heads:
  - Backbone-frozen pipeline: input is precomputed 768-d CLS vector x.
  - concept_adapter: MLP(768 -> 256 -> K_concepts) producing concept logits c.
  - class heads: PER-COHORT heads classifier_u: MLP(c -> num_classes_u), so each
    cohort's class taxonomy is honoured without forcing label-space alignment.
  - cohort_head: MLP(c -> 3) — the iVAE auxiliary head that maps concept output
    to cohort ID. This is the supervised auxiliary-variable conditioning of
    Khemakhem 2020.

Loss:
  L = w_class * sum_u L_class_u(classifier_u(c), y_u)         (CE)
    + w_concept * L_concept(c, c_target, supervised_mask)     (BCE, masked)
    + w_aux * L_aux(cohort_head(c), u)                        (CE on cohort ID)

The aux head pushes c to be cohort-separable; theoretically (Khemakhem 2020),
this restores per-cell concept identifiability up to permutation + scaling
provided cohort distributions are sufficiently distinct.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

WORKDIR = Path("/gpfs/workdir/mouaddenn")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Use the same concept-matrix utilities as train.py
from models import build_prior_C, build_class_concept_targets, aggregate_concept_target


# ============================================================================
# Cohort data loaders (cache-backed)
# ============================================================================

GR_FEAT_NPZ = "/gpfs/workdir/mouaddenn/thesis/ch3_gr_neutro/outputs/dinobloom_features.npz"
GR_ANN = "/gpfs/workdir/mouaddenn/data/gr_neutro_extended/annotations.csv"
AML_FEAT_NPZ = "/gpfs/workdir/mouaddenn/data/aml_matek/features/dinobloom_b_cls.npz"
MLL_FEAT_NPZ = "/gpfs/workdir/mouaddenn/data/mll23/mll23_dinobloom_features.npz"

# AML Matek class index -> GR-Neutro class index for the class->concept matrix.
# Only mature-neutrophil indices map cleanly. Others get no concept supervision.
# Matek class space (from cache_features.py):
#  0 MYB, 1 PMO/PMB, 2 MYO, 3 MMZ, 4 BAS, 5 EOS, 6 NGB, 7 NGS, 8 MON, 9 LYT,
# 11 EBO, 13 LYA/MOB, 14 KSC
AML_TO_GR_CONCEPT_CLASS = {
    7: 0,  # NGS (segmented neutrophil)        -> GR-Neutro Normal
    6: 6,  # NGB (band neutrophil)             -> GR-Neutro Hyposegmentation
}

# MLL-23 source label -> GR-Neutro class index (concept supervision only when known).
# Same convention as MLL-23 map.
MLL_TO_GR_CONCEPT_CLASS = {
    "neutrophil_segmented": 0,  # -> Normal
    "neutrophil_band": 6,       # -> Hyposegmentation
}


def load_gr_neutro(class_concept: torch.Tensor, normal_idx: int):
    """Return (features [N,768], class_idx [N] long {GR-Neutro 7-cls},
              concept_target [N, K] in [0,1], concept_mask [N] bool).
    Class supervision available (one-hot top-active class from annotations).
    Concept supervision available for ALL cells.
    """
    d = np.load(GR_FEAT_NPZ, allow_pickle=True)
    feats = torch.from_numpy(d["features"]).float()  # (N, 768)
    paths = d["paths"]
    # Map path basename -> annotation row.
    ann = {}
    with open(GR_ANN) as f:
        r = csv.reader(f); hdr = next(r)
        class_names = hdr[2:]
        for row in r:
            ann[row[0]] = [int(x) for x in row[2:]]
    labels = np.zeros((len(paths), len(class_names)), dtype=np.float32)
    for i, p in enumerate(paths):
        bn = Path(p).name
        labels[i] = ann[bn]
    multilabel = torch.from_numpy(labels)
    # Concept targets via max-deviation aggregation (same as train.py).
    concept_target = aggregate_concept_target(class_concept, multilabel,
                                              normal_idx=normal_idx)  # (N, K) in [0,1]
    # Class index = first active label (multi-label cells are rare; pick min idx).
    # Per train.py convention we use the multi-label vector for losses, but for
    # multicohort CE we collapse to a single class for cohort-coherent training.
    cls_idx = labels.argmax(axis=1).astype(np.int64)
    # Concept mask: True for all GR-Neutro cells.
    concept_mask = np.ones(len(paths), dtype=bool)
    return feats, torch.from_numpy(cls_idx), concept_target, torch.from_numpy(concept_mask), class_names, paths


def load_aml_matek(class_concept: torch.Tensor, normal_idx: int):
    """AML Matek 2019 cohort. Class label is multi-class (15 classes). Concept
    supervision: only for cells whose Matek class is in AML_TO_GR_CONCEPT_CLASS;
    others get a zero target with mask=False.
    """
    if not Path(AML_FEAT_NPZ).exists():
        print(f"[warn] {AML_FEAT_NPZ} missing — caching not yet complete?", flush=True)
        return None
    d = np.load(AML_FEAT_NPZ, allow_pickle=True)
    feats = torch.from_numpy(d["features"]).float()
    labels = torch.from_numpy(d["labels"]).long()
    N = feats.shape[0]
    K = class_concept.shape[1]
    concept_target = torch.zeros(N, K, dtype=torch.float32)
    concept_mask = np.zeros(N, dtype=bool)
    for i, lbl in enumerate(labels.tolist()):
        gr_cls = AML_TO_GR_CONCEPT_CLASS.get(int(lbl))
        if gr_cls is not None:
            # Use the GR-Neutro class row as the concept target.
            concept_target[i] = class_concept[gr_cls]
            concept_mask[i] = True
    print(f"[aml] {N} cells; concept-supervised: {concept_mask.sum()}", flush=True)
    return feats, labels, concept_target, torch.from_numpy(concept_mask)


def load_mll23(class_concept: torch.Tensor, normal_idx: int):
    """MLL-23 cohort: 768-d DinoBloom-B features. Class space here is the
    18-class MLL-23 taxonomy; we encode it as int labels 0..17 alphabetically.
    Concept supervision: only neutrophil_segmented + neutrophil_band, via map.
    """
    d = np.load(MLL_FEAT_NPZ, allow_pickle=True)
    feats = torch.from_numpy(d["features"]).float()
    sources = d["sources"]  # array of class names
    unique_sources = sorted(set(sources.tolist()))
    src_to_idx = {s: i for i, s in enumerate(unique_sources)}
    labels = torch.tensor([src_to_idx[s] for s in sources.tolist()], dtype=torch.long)
    N = feats.shape[0]
    K = class_concept.shape[1]
    concept_target = torch.zeros(N, K, dtype=torch.float32)
    concept_mask = np.zeros(N, dtype=bool)
    for i, s in enumerate(sources.tolist()):
        gr_cls = MLL_TO_GR_CONCEPT_CLASS.get(s)
        if gr_cls is not None:
            concept_target[i] = class_concept[gr_cls]
            concept_mask[i] = True
    print(f"[mll] {N} cells; {len(unique_sources)} classes; "
          f"concept-supervised: {concept_mask.sum()}", flush=True)
    return feats, labels, concept_target, torch.from_numpy(concept_mask), unique_sources


# ============================================================================
# Dataset wrapping the joined tensors
# ============================================================================

class MultiCohortDataset(Dataset):
    def __init__(self, feats, cls_idx, concept_target, concept_mask, cohort_id):
        self.feats = feats
        self.cls_idx = cls_idx
        self.concept_target = concept_target
        self.concept_mask = concept_mask
        self.cohort_id = cohort_id  # int

    def __len__(self): return self.feats.shape[0]
    def __getitem__(self, i):
        return (self.feats[i], self.cls_idx[i], self.concept_target[i],
                self.concept_mask[i].item(), self.cohort_id)


def split_indices(N, seed, val_frac=0.10, test_frac=0.10):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(N)
    n_test = int(round(N * test_frac))
    n_val = int(round(N * val_frac))
    test = perm[:n_test]; val = perm[n_test:n_test+n_val]
    tr = perm[n_test+n_val:]
    return tr, val, test


# ============================================================================
# Model: concept adapter + per-cohort classifier + cohort head
# ============================================================================

class MultiCohortCBM(nn.Module):
    def __init__(self, embed_dim, K_concepts, cohort_num_classes: dict[int, int],
                 num_cohorts: int, hidden=256, dropout=0.3):
        super().__init__()
        self.concept_net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, K_concepts),
        )
        # Per-cohort class heads on top of concept logits.
        self.cohort_heads = nn.ModuleDict()
        for u, ncls in cohort_num_classes.items():
            self.cohort_heads[str(u)] = nn.Sequential(
                nn.Dropout(dropout * 0.5),
                nn.Linear(K_concepts, ncls),
            )
        # Auxiliary cohort prediction head: maps c -> cohort id u.
        self.cohort_aux = nn.Sequential(
            nn.LayerNorm(K_concepts),
            nn.Linear(K_concepts, num_cohorts),
        )

    def forward(self, x, u):
        """x: (B, D)  u: (B,) long cohort ids.
        Returns: concept_logits (B, K), class_logits (B, max_ncls_for_each_u),
                 cohort_logits (B, U). Class logits per row are computed by the
                 head for that row's cohort.
        """
        c = self.concept_net(x)
        cohort_logits = self.cohort_aux(c)
        return c, cohort_logits


# ============================================================================
# Training loop
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--w_class", type=float, default=1.0)
    ap.add_argument("--w_concept", type=float, default=2.0)
    ap.add_argument("--w_aux", type=float, default=0.5,
                    help="Weight on cohort-ID auxiliary head loss (iVAE conditioning).")
    ap.add_argument("--concept_target_mode", default="hard", choices=["soft", "hard"])
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--out_root", default=str(HERE / "outputs" / "multi_cohort"))
    ap.add_argument("--config", default=str(HERE / "concept_config_gr_neutro.json"))
    ap.add_argument("--use_aml", action="store_true", default=True)
    ap.add_argument("--no_aml", dest="use_aml", action="store_false")
    ap.add_argument("--use_mll", action="store_true", default=True)
    ap.add_argument("--no_mll", dest="use_mll", action="store_false")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}  [tag] {args.tag}  [seed] {args.seed}", flush=True)
    out_dir = Path(args.out_root) / f"{args.tag}_s{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- concept config ----------
    cfg = json.loads(Path(args.config).read_text())
    concepts = cfg["concepts"]
    K_concepts = len(concepts)
    class_concept = build_class_concept_targets(cfg["class_to_concept_matrix"]["matrix"])
    prior_C = build_prior_C(concepts, cfg["concept_constraint_matrix"])
    normal_idx = int(cfg.get("normal_class_index", 0))
    K_gr = class_concept.shape[0]

    # ---------- load cohorts ----------
    gr_feats, gr_cls, gr_ct, gr_mask, gr_class_names, gr_paths = load_gr_neutro(
        class_concept, normal_idx)
    print(f"[gr] N={len(gr_feats)} classes={gr_class_names}", flush=True)
    cohort_data = {0: (gr_feats, gr_cls, gr_ct, gr_mask, K_gr)}
    cohort_names = {0: "GR-Neutro"}

    if args.use_aml:
        aml = load_aml_matek(class_concept, normal_idx)
        if aml is not None:
            aml_feats, aml_cls, aml_ct, aml_mask = aml
            cohort_data[1] = (aml_feats, aml_cls, aml_ct, aml_mask, int(aml_cls.max().item()) + 1)
            cohort_names[1] = "AML-Matek"
        else:
            print("[warn] AML Matek B-features not yet available; training without AML", flush=True)

    mll_class_names = None
    if args.use_mll:
        mll_feats, mll_cls, mll_ct, mll_mask, mll_class_names = load_mll23(
            class_concept, normal_idx)
        cohort_data[2] = (mll_feats, mll_cls, mll_ct, mll_mask,
                          int(mll_cls.max().item()) + 1)
        cohort_names[2] = "MLL-23"

    cohorts_used = sorted(cohort_data.keys())
    num_cohorts = len(cohorts_used)
    print(f"[cohorts] using {cohorts_used} -> {[cohort_names[u] for u in cohorts_used]}",
          flush=True)
    # Compact remap so cohort_aux head has contiguous output ids.
    u_remap = {u: i for i, u in enumerate(cohorts_used)}

    # ---------- per-cohort split + dataset ----------
    train_datasets, val_datasets, test_datasets = {}, {}, {}
    cohort_num_classes = {}
    for u, (feats, cls, ct, mask, ncls) in cohort_data.items():
        tr, val, te = split_indices(len(feats), seed=args.seed)
        train_datasets[u] = MultiCohortDataset(
            feats[tr], cls[tr], ct[tr], mask[tr], u_remap[u])
        val_datasets[u] = MultiCohortDataset(
            feats[val], cls[val], ct[val], mask[val], u_remap[u])
        test_datasets[u] = MultiCohortDataset(
            feats[te], cls[te], ct[te], mask[te], u_remap[u])
        cohort_num_classes[u_remap[u]] = ncls
        print(f"[{cohort_names[u]}] train={len(tr)} val={len(val)} test={len(te)} ncls={ncls}",
              flush=True)
        # Save full feats + splits per cohort for downstream eval
        torch.save({
            "feats_all": feats, "cls_all": cls, "ct_all": ct, "mask_all": mask,
            "train_idx": torch.tensor(tr), "val_idx": torch.tensor(val),
            "test_idx": torch.tensor(te), "cohort_name": cohort_names[u],
        }, out_dir / f"cohort_{u_remap[u]}_split.pt")

    train_concat = torch.utils.data.ConcatDataset(
        [train_datasets[u] for u in cohorts_used])

    # ---------- model ----------
    embed_dim = gr_feats.shape[1]
    model = MultiCohortCBM(
        embed_dim=embed_dim, K_concepts=K_concepts,
        cohort_num_classes=cohort_num_classes, num_cohorts=num_cohorts,
        hidden=args.hidden, dropout=args.dropout,
    ).to(device)
    print(f"[model] embed={embed_dim} K={K_concepts} cohorts={num_cohorts} "
          f"per-cohort ncls={cohort_num_classes}", flush=True)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    # ---------- loss helpers ----------
    bce_logits = nn.BCEWithLogitsLoss(reduction="none")
    ce_logits = nn.CrossEntropyLoss(reduction="mean")

    def cohort_class_loss(c_logits, u_remapped, cls_idx):
        """Apply per-cohort head and compute CE only for rows from each cohort."""
        total = 0.0; n = 0
        per_u_logits = {}
        for u_re in cohort_num_classes.keys():
            mask_u = (u_remapped == u_re)
            if not mask_u.any():
                continue
            head = model.cohort_heads[str(u_re)]
            logits_u = head(c_logits[mask_u])
            per_u_logits[u_re] = (mask_u, logits_u)
            ce = ce_logits(logits_u, cls_idx[mask_u])
            total = total + ce * mask_u.sum()
            n += mask_u.sum().item()
        return (total / max(n, 1)) if n > 0 else c_logits.sum() * 0.0, per_u_logits

    train_loader = DataLoader(train_concat, batch_size=args.batch_size,
                              shuffle=True, num_workers=2, drop_last=False)
    log_path = out_dir / "training_log.csv"
    with open(log_path, "w", newline="") as fh:
        csv.writer(fh).writerow([
            "epoch", "L", "L_class", "L_concept", "L_aux",
            "val_gr_acc", "val_aml_acc", "val_mll_acc", "val_concept_f1", "lr",
        ])

    def evaluate(dsets):
        model.eval()
        out = {}
        all_concept_probs, all_concept_targ, all_concept_mask = [], [], []
        with torch.no_grad():
            for u, ds in dsets.items():
                if len(ds) == 0: continue
                u_re = u_remap[u]
                loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)
                cls_correct = 0; n = 0
                all_clp, all_cls_true = [], []
                for x, ci, ct, mk, u_id in loader:
                    x = x.to(device); ci = ci.to(device); ct = ct.to(device)
                    c, _ = model(x, u_id.to(device))
                    head = model.cohort_heads[str(u_re)]
                    cls_logits = head(c)
                    pred = cls_logits.argmax(dim=1)
                    cls_correct += (pred == ci).sum().item(); n += x.size(0)
                    if u == 0:
                        all_concept_probs.append(c.sigmoid().cpu())
                        all_concept_targ.append(ct.cpu())
                        all_concept_mask.append(mk.cpu())
                    all_clp.append(cls_logits.softmax(dim=1).cpu())
                    all_cls_true.append(ci.cpu())
                out[u] = dict(acc=cls_correct/max(n,1), n=n,
                              cls_probs=torch.cat(all_clp).numpy(),
                              cls_true=torch.cat(all_cls_true).numpy())
        # Concept F1 on GR-Neutro test set (the rank-decompression target).
        if all_concept_probs:
            cp = torch.cat(all_concept_probs).numpy()
            ctg = torch.cat(all_concept_targ).numpy()
            cmk = torch.cat(all_concept_mask).numpy()
            # Hard targets
            y = (ctg >= 0.5).astype(int)
            pr = (cp >= 0.5).astype(int)
            from sklearn.metrics import f1_score
            f1 = f1_score(y[cmk.astype(bool)], pr[cmk.astype(bool)],
                          average="macro", zero_division=0)
            out["_concept_f1_gr"] = float(f1)
            out["_concept_probs_gr"] = cp
            out["_concept_targets_gr"] = ctg
        return out

    # ---------- training ----------
    t0 = time.time()
    best_score = -1.0; best_state = None
    for ep in range(args.epochs):
        model.train()
        n_tot = 0
        L_tot = L_cls_tot = L_con_tot = L_aux_tot = 0.0
        for x, ci, ct, mk, u_id in train_loader:
            x = x.to(device); ci = ci.to(device); ct = ct.to(device)
            mk = mk.to(device); u_id = u_id.to(device)
            c, u_logits = model(x, u_id)
            # Class loss (per-cohort heads).
            L_cls, _ = cohort_class_loss(c, u_id, ci)
            # Concept BCE — masked.
            if mk.any():
                tgt = (ct[mk] >= 0.5).float() if args.concept_target_mode == "hard" else ct[mk]
                L_con = bce_logits(c[mk], tgt).mean()
            else:
                L_con = c.sum() * 0.0
            # Aux cohort-id loss (iVAE conditioning).
            L_aux = ce_logits(u_logits, u_id)
            L = args.w_class * L_cls + args.w_concept * L_con + args.w_aux * L_aux
            optim.zero_grad(set_to_none=True); L.backward(); optim.step()
            B = x.size(0); n_tot += B
            L_tot += L.item() * B; L_cls_tot += L_cls.item() * B
            L_con_tot += float(L_con) * B; L_aux_tot += L_aux.item() * B
        sched.step()
        val = evaluate(val_datasets)
        v_gr = val.get(0, {}).get("acc", 0.0)
        v_am = val.get(1, {}).get("acc", 0.0)
        v_ml = val.get(2, {}).get("acc", 0.0)
        v_cf = val.get("_concept_f1_gr", 0.0)
        with open(log_path, "a", newline="") as fh:
            csv.writer(fh).writerow([
                ep, f"{L_tot/n_tot:.4f}", f"{L_cls_tot/n_tot:.4f}",
                f"{L_con_tot/n_tot:.4f}", f"{L_aux_tot/n_tot:.4f}",
                f"{v_gr:.4f}", f"{v_am:.4f}", f"{v_ml:.4f}",
                f"{v_cf:.4f}", f"{optim.param_groups[0]['lr']:.6f}",
            ])
        score = v_gr + v_cf  # rank by GR-Neutro acc + concept F1
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"[ep {ep:3d}] L={L_tot/n_tot:.3f} cls={L_cls_tot/n_tot:.3f} "
                  f"con={L_con_tot/n_tot:.3f} aux={L_aux_tot/n_tot:.3f} | "
                  f"v_gr={v_gr:.3f} v_aml={v_am:.3f} v_mll={v_ml:.3f} "
                  f"v_cf={v_cf:.3f} ({time.time()-t0:.0f}s)", flush=True)

    print(f"[done] training: {time.time()-t0:.0f}s, best score={best_score:.3f}",
          flush=True)
    model.load_state_dict(best_state)

    # ---------- final test eval + per-cohort dump ----------
    test = evaluate(test_datasets)
    # Compute on FULL data per cohort for downstream identifiability analysis.
    model.eval()
    full_concept_probs = {}
    full_class_logits = {}
    full_cls_true = {}
    full_aux_logits = {}
    with torch.no_grad():
        for u, (feats, cls, ct, mk, ncls) in cohort_data.items():
            u_re = u_remap[u]
            ds = MultiCohortDataset(feats, cls, ct, mk, u_re)
            loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)
            all_c, all_u, all_cls_logits, all_cls_true = [], [], [], []
            for x, ci, ctt, mki, u_id in loader:
                x = x.to(device)
                c, u_logits = model(x, u_id.to(device))
                head = model.cohort_heads[str(u_re)]
                cls_logits = head(c)
                all_c.append(c.sigmoid().cpu())
                all_u.append(u_logits.softmax(dim=1).cpu())
                all_cls_logits.append(cls_logits.cpu())
                all_cls_true.append(ci.cpu())
            full_concept_probs[u] = torch.cat(all_c).numpy()
            full_aux_logits[u] = torch.cat(all_u).numpy()
            full_class_logits[u] = torch.cat(all_cls_logits).numpy()
            full_cls_true[u] = torch.cat(all_cls_true).numpy()

    # Per-seed predictions dump
    pred_path = out_dir.parent / f"predictions_s{args.seed}.pt"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "concepts": concepts, "K_concepts": K_concepts,
        "cohorts_used": cohorts_used, "cohort_names": cohort_names,
        "u_remap": u_remap, "gr_class_names": gr_class_names,
        "mll_class_names": mll_class_names,
        "full_concept_probs": {u: full_concept_probs[u] for u in cohorts_used},
        "full_class_logits": {u: full_class_logits[u] for u in cohorts_used},
        "full_cls_true": {u: full_cls_true[u] for u in cohorts_used},
        "full_aux_probs": {u: full_aux_logits[u] for u in cohorts_used},
        "gr_paths": gr_paths,
        "splits": {u: {
            "train_idx": split_indices(len(cohort_data[u][0]), args.seed)[0].tolist(),
            "val_idx":   split_indices(len(cohort_data[u][0]), args.seed)[1].tolist(),
            "test_idx":  split_indices(len(cohort_data[u][0]), args.seed)[2].tolist(),
        } for u in cohorts_used},
        "args": vars(args),
        "best_state": best_state,
    }, pred_path)
    print(f"[save] {pred_path}", flush=True)

    # Test summary
    from sklearn.metrics import f1_score
    # GR-Neutro test W-F1
    gr_test_idx = split_indices(len(gr_feats), args.seed)[2]
    gr_test_pred = full_class_logits[0][gr_test_idx].argmax(axis=1)
    gr_test_true = gr_cls.numpy()[gr_test_idx]
    gr_wf1 = float(f1_score(gr_test_true, gr_test_pred, average="weighted",
                            zero_division=0))
    gr_mf1 = float(f1_score(gr_test_true, gr_test_pred, average="macro",
                            zero_division=0))

    # MLL-23 Normal recall (cells with source neutrophil_segmented -> idx in
    # remapped 18-class label space).
    mll_normal_recall = float("nan")
    if mll_class_names is not None:
        try:
            normal_id = mll_class_names.index("neutrophil_segmented")
            mll_true = full_cls_true[2]
            mll_pred = full_class_logits[2].argmax(axis=1)
            mask = (mll_true == normal_id)
            mll_normal_recall = float((mll_pred[mask] == normal_id).mean())
        except Exception as e:
            print(f"[warn] MLL normal recall: {e}", flush=True)

    # Concept identifiability: singular values of concept-prob matrix on GR-Neutro test.
    cp_gr_test = full_concept_probs[0][gr_test_idx]  # (n_test, K)
    cp_centered = cp_gr_test - cp_gr_test.mean(axis=0, keepdims=True)
    s = np.linalg.svd(cp_centered, compute_uv=False)
    s_norm = s / s.max() if s.max() > 0 else s
    n_above = int((s_norm >= 0.05).sum())
    gap_7_8 = float(s_norm[6] - s_norm[7]) if len(s_norm) >= 8 else float("nan")

    print(f"\n[summary] GR-Neutro test W-F1={gr_wf1:.4f}  Macro-F1={gr_mf1:.4f}", flush=True)
    print(f"[summary] MLL-23 Normal recall={mll_normal_recall:.4f}", flush=True)
    print(f"[summary] concept SV (normalized, GR test): {s_norm.round(3).tolist()}", flush=True)
    print(f"[summary] n_above(0.05)={n_above}, gap(s7-s8)={gap_7_8:.4f}", flush=True)

    summary = dict(
        tag=args.tag, seed=args.seed, time_seconds=time.time() - t0,
        cohorts_used=cohorts_used,
        cohort_names=cohort_names,
        gr_test_weighted_f1=gr_wf1, gr_test_macro_f1=gr_mf1,
        mll_normal_recall=mll_normal_recall,
        concept_singular_values_normalized=s_norm.tolist(),
        n_concepts_above_0p05=n_above,
        gap_sigma7_sigma8=gap_7_8,
        args=vars(args),
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[save] {out_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
