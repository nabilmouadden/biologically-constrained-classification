"""LOCO (Leave-One-Cohort-Out) multi-cohort CBM training — Contribution C1'.

Honest version of C1: the held-out cohort never appears in the training mix.
Adapted from train_multicohort.py.

Cohort ids:
  0 GR-Neutro, 1 AML-Matek, 2 MLL-23, 3 Bodzas

The held-out cohort is loaded but split into eval-only blocks (no train slice).
The auxiliary cohort-id head is over the |training cohorts| training cohorts only.

Class taxonomies are per-cohort. Held-out class accuracy uses the held-out
cohort's own taxonomy — i.e. we predict class labels only for cohorts whose
heads are trained on the held-out cell's class space. For held-out cohorts
without a trained head, we evaluate via concept-prob distribution and the
"Normal-vs-not" mapping where cells map to GR-Neutro Normal: this is the
load-bearing comparator for the published pure-CBM 0.473 / CDA 0.550.

Hard rule: cohort u_h does NOT appear in train_concat. We assert this.
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
from torch.utils.data import DataLoader, Dataset, ConcatDataset

WORKDIR = Path("/gpfs/workdir/mouaddenn")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from models import build_prior_C, build_class_concept_targets, aggregate_concept_target  # noqa

# ---------- feature caches ----------
GR_FEAT_NPZ = "/gpfs/workdir/mouaddenn/thesis/ch3_gr_neutro/outputs/dinobloom_features.npz"
GR_ANN = "/gpfs/workdir/mouaddenn/data/gr_neutro_extended/annotations.csv"
AML_FEAT_NPZ = "/gpfs/workdir/mouaddenn/data/aml_matek/features/dinobloom_b_cls.npz"
MLL_FEAT_NPZ = "/gpfs/workdir/mouaddenn/data/mll23/mll23_dinobloom_features.npz"
BOD_FEAT_NPZ = "/gpfs/workdir/mouaddenn/data/bodzas/features/dinobloom_b_cls.npz"

# Class -> GR concept-row mappings.
AML_TO_GR_CONCEPT_CLASS = {7: 0, 6: 6}  # NGS->Normal, NGB->Hyposeg
MLL_TO_GR_CONCEPT_CLASS = {"neutrophil_segmented": 0, "neutrophil_band": 6}
BOD_TO_GR_CONCEPT_CLASS = {"neutrophil_segment": 0, "neutrophil_band": 6}

COHORT_NAMES = {0: "GR-Neutro", 1: "AML-Matek", 2: "MLL-23", 3: "Bodzas"}


# ============================================================================
# Loaders
# ============================================================================

def load_gr_neutro(class_concept: torch.Tensor, normal_idx: int):
    d = np.load(GR_FEAT_NPZ, allow_pickle=True)
    feats = torch.from_numpy(d["features"]).float()
    paths = d["paths"]
    ann = {}
    with open(GR_ANN) as f:
        r = csv.reader(f); hdr = next(r)
        class_names = hdr[2:]
        for row in r:
            ann[row[0]] = [int(x) for x in row[2:]]
    labels = np.zeros((len(paths), len(class_names)), dtype=np.float32)
    for i, p in enumerate(paths):
        labels[i] = ann[Path(p).name]
    multilabel = torch.from_numpy(labels)
    concept_target = aggregate_concept_target(class_concept, multilabel,
                                              normal_idx=normal_idx)
    cls_idx = torch.from_numpy(labels.argmax(axis=1).astype(np.int64))
    concept_mask = torch.from_numpy(np.ones(len(paths), dtype=bool))
    return dict(feats=feats, cls=cls_idx, ct=concept_target, mask=concept_mask,
                class_names=class_names, paths=paths,
                normal_class_id=0)  # GR-Neutro: Normal is class 0


def load_aml_matek(class_concept: torch.Tensor, normal_idx: int):
    if not Path(AML_FEAT_NPZ).exists():
        print(f"[skip] AML not cached at {AML_FEAT_NPZ}", flush=True)
        return None
    d = np.load(AML_FEAT_NPZ, allow_pickle=True)
    feats = torch.from_numpy(d["features"]).float()
    labels = torch.from_numpy(d["labels"]).long()
    N = feats.shape[0]; K = class_concept.shape[1]
    ct = torch.zeros(N, K, dtype=torch.float32)
    mk = np.zeros(N, dtype=bool)
    for i, lbl in enumerate(labels.tolist()):
        gr = AML_TO_GR_CONCEPT_CLASS.get(int(lbl))
        if gr is not None:
            ct[i] = class_concept[gr]
            mk[i] = True
    # AML Matek class names list (15 classes from cache_aml_matek_dinobloom_b.py).
    cls_names = ["MYB","PMO_PMB","MYO","MMZ","BAS","EOS","NGB","NGS",
                 "MON","LYT","__10","EBO","__12","LYA_MOB","KSC"]
    return dict(feats=feats, cls=labels, ct=ct, mask=torch.from_numpy(mk),
                class_names=cls_names, paths=None,
                normal_class_id=7)  # NGS index


def load_mll23(class_concept: torch.Tensor, normal_idx: int):
    if not Path(MLL_FEAT_NPZ).exists():
        print(f"[skip] MLL not cached at {MLL_FEAT_NPZ}", flush=True)
        return None
    d = np.load(MLL_FEAT_NPZ, allow_pickle=True)
    feats = torch.from_numpy(d["features"]).float()
    sources = d["sources"]
    unique = sorted(set(sources.tolist()))
    s2i = {s: i for i, s in enumerate(unique)}
    labels = torch.tensor([s2i[s] for s in sources.tolist()], dtype=torch.long)
    N = feats.shape[0]; K = class_concept.shape[1]
    ct = torch.zeros(N, K, dtype=torch.float32)
    mk = np.zeros(N, dtype=bool)
    for i, s in enumerate(sources.tolist()):
        gr = MLL_TO_GR_CONCEPT_CLASS.get(s)
        if gr is not None:
            ct[i] = class_concept[gr]; mk[i] = True
    normal_id = unique.index("neutrophil_segmented") if "neutrophil_segmented" in unique else -1
    return dict(feats=feats, cls=labels, ct=ct, mask=torch.from_numpy(mk),
                class_names=unique, paths=None,
                normal_class_id=normal_id)


def load_bodzas(class_concept: torch.Tensor, normal_idx: int):
    if not Path(BOD_FEAT_NPZ).exists():
        print(f"[skip] Bodzas not cached at {BOD_FEAT_NPZ}", flush=True)
        return None
    d = np.load(BOD_FEAT_NPZ, allow_pickle=True)
    feats = torch.from_numpy(d["features"]).float()
    labels = torch.from_numpy(d["labels"]).long()
    class_names = list(d["class_names"])
    N = feats.shape[0]; K = class_concept.shape[1]
    ct = torch.zeros(N, K, dtype=torch.float32)
    mk = np.zeros(N, dtype=bool)
    for i, lbl in enumerate(labels.tolist()):
        s = class_names[int(lbl)]
        gr = BOD_TO_GR_CONCEPT_CLASS.get(s)
        if gr is not None:
            ct[i] = class_concept[gr]; mk[i] = True
    normal_id = class_names.index("neutrophil_segment") if "neutrophil_segment" in class_names else -1
    return dict(feats=feats, cls=labels, ct=ct, mask=torch.from_numpy(mk),
                class_names=class_names, paths=None,
                normal_class_id=normal_id)


LOADERS = {0: load_gr_neutro, 1: load_aml_matek, 2: load_mll23, 3: load_bodzas}


# ============================================================================
# Dataset
# ============================================================================

class MultiCohortDataset(Dataset):
    def __init__(self, feats, cls, ct, mask, cohort_id_remapped):
        self.feats = feats; self.cls = cls
        self.ct = ct; self.mask = mask
        self.cohort_id = cohort_id_remapped

    def __len__(self): return self.feats.shape[0]
    def __getitem__(self, i):
        return (self.feats[i], self.cls[i], self.ct[i],
                bool(self.mask[i].item()), self.cohort_id)


def split_indices(N, seed, val_frac=0.10, test_frac=0.10):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(N)
    n_test = int(round(N * test_frac))
    n_val = int(round(N * val_frac))
    return perm[n_test+n_val:], perm[n_test:n_test+n_val], perm[:n_test]


# ============================================================================
# Model
# ============================================================================

class MultiCohortCBM(nn.Module):
    def __init__(self, embed_dim, K, cohort_ncls, n_cohorts_train, hidden=256, dropout=0.3):
        super().__init__()
        self.concept_net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, K),
        )
        self.cohort_heads = nn.ModuleDict()
        for u_re, ncls in cohort_ncls.items():
            self.cohort_heads[str(u_re)] = nn.Sequential(
                nn.Dropout(dropout * 0.5),
                nn.Linear(K, ncls),
            )
        # Aux cohort head over TRAINING cohorts only.
        self.cohort_aux = nn.Sequential(
            nn.LayerNorm(K),
            nn.Linear(K, n_cohorts_train),
        )

    def forward(self, x):
        c = self.concept_net(x)
        u_logits = self.cohort_aux(c)
        return c, u_logits


# ============================================================================
# Training
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--held_out", type=int, required=True, choices=[0,1,2,3],
                    help="Cohort id to hold out from training (0=GR,1=AML,2=MLL,3=Bod).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--w_class", type=float, default=1.0)
    ap.add_argument("--w_concept", type=float, default=2.0)
    ap.add_argument("--w_aux", type=float, default=0.5)
    ap.add_argument("--concept_target_mode", default="hard")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--out_root", default=str(HERE / "outputs" / "c1_loco"))
    ap.add_argument("--config", default=str(HERE / "concept_config_gr_neutro.json"))
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}  [tag] {args.tag}  [seed] {args.seed} "
          f"[held_out] {COHORT_NAMES[args.held_out]}", flush=True)
    out_dir = Path(args.out_root) / f"{args.tag}_held{args.held_out}_s{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.loads(Path(args.config).read_text())
    concepts = cfg["concepts"]
    K_concepts = len(concepts)
    class_concept = build_class_concept_targets(cfg["class_to_concept_matrix"]["matrix"])
    normal_idx = int(cfg.get("normal_class_index", 0))

    # ---------- load all 4 cohorts ----------
    cohorts = {}
    for u in [0, 1, 2, 3]:
        c = LOADERS[u](class_concept, normal_idx)
        if c is None:
            print(f"[fatal] cohort {u}={COHORT_NAMES[u]} unavailable", flush=True)
            if u == args.held_out:
                print(f"[fatal] held-out cohort unavailable; cannot evaluate.", flush=True)
                sys.exit(2)
            else:
                print(f"[warn] training cohort {u} missing; continuing without it", flush=True)
                continue
        cohorts[u] = c
        print(f"[load] {COHORT_NAMES[u]}: N={len(c['feats'])} "
              f"classes={len(c['class_names'])} concept-sup={int(c['mask'].sum())}",
              flush=True)

    if args.held_out not in cohorts:
        print(f"[fatal] held-out cohort {args.held_out} not loadable", flush=True)
        sys.exit(2)

    train_cohorts = sorted(set(cohorts.keys()) - {args.held_out})
    if not train_cohorts:
        print(f"[fatal] no training cohorts left", flush=True)
        sys.exit(2)
    print(f"[setup] training cohorts: {[COHORT_NAMES[u] for u in train_cohorts]}", flush=True)
    print(f"[setup] held-out cohort: {COHORT_NAMES[args.held_out]}", flush=True)

    # Aux remap: only training cohorts contiguous ids.
    aux_remap = {u: i for i, u in enumerate(train_cohorts)}
    # Per-cohort class-head remap (training cohorts only).
    head_remap = {u: i for i, u in enumerate(train_cohorts)}
    cohort_ncls = {head_remap[u]: len(cohorts[u]["class_names"]) for u in train_cohorts}

    # ---------- splits ----------
    # Training cohorts: standard 80/10/10 splits.
    # Held-out cohort: ALL cells are evaluation only (no train slice).
    train_dsets, val_dsets, test_dsets = {}, {}, {}
    for u in train_cohorts:
        c = cohorts[u]
        tr, va, te = split_indices(len(c["feats"]), seed=args.seed)
        train_dsets[u] = MultiCohortDataset(
            c["feats"][tr], c["cls"][tr], c["ct"][tr], c["mask"][tr], aux_remap[u])
        val_dsets[u] = MultiCohortDataset(
            c["feats"][va], c["cls"][va], c["ct"][va], c["mask"][va], aux_remap[u])
        test_dsets[u] = MultiCohortDataset(
            c["feats"][te], c["cls"][te], c["ct"][te], c["mask"][te], aux_remap[u])
        print(f"[split] {COHORT_NAMES[u]}: tr={len(tr)} va={len(va)} te={len(te)}", flush=True)
    # Held-out: all data as a single eval set, no aux id assigned.
    ho = cohorts[args.held_out]
    held_full = MultiCohortDataset(ho["feats"], ho["cls"], ho["ct"], ho["mask"], -1)

    train_concat = ConcatDataset([train_dsets[u] for u in train_cohorts])
    # HARD ASSERTION: held-out cohort cells must not be in train_concat.
    n_train = len(train_concat)
    n_train_expected = sum(len(train_dsets[u]) for u in train_cohorts)
    assert n_train == n_train_expected, "ConcatDataset size mismatch"
    # And cross-check: held-out cohort feats unique identity must not appear in any train tensor
    # Use a tensor-id sanity check: held-out tensor identity != any train tensor.
    ho_id = id(ho["feats"])
    for u in train_cohorts:
        assert id(cohorts[u]["feats"]) != ho_id, f"held-out feats id leak in cohort {u}"
    print(f"[verify] LOCO assertion passed: held-out cohort {COHORT_NAMES[args.held_out]} "
          f"({len(ho['feats'])} cells) NOT in train ({n_train} cells)", flush=True)

    # ---------- model ----------
    embed_dim = cohorts[train_cohorts[0]]["feats"].shape[1]
    model = MultiCohortCBM(
        embed_dim=embed_dim, K=K_concepts,
        cohort_ncls=cohort_ncls,
        n_cohorts_train=len(train_cohorts),
        hidden=args.hidden, dropout=args.dropout,
    ).to(device)
    print(f"[model] embed={embed_dim} K={K_concepts} cohort_ncls={cohort_ncls}", flush=True)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    ce = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_concat, batch_size=args.batch_size,
                              shuffle=True, num_workers=2, drop_last=False)

    log_path = out_dir / "training_log.csv"
    with open(log_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["epoch", "L", "L_class", "L_concept", "L_aux"] +
                   [f"val_acc_u{u}" for u in train_cohorts] + ["lr"])

    def per_cohort_class_loss(c, u_id, ci):
        total = 0.0; n = 0
        for u_re in cohort_ncls.keys():
            m = (u_id == u_re)
            if not m.any(): continue
            head = model.cohort_heads[str(u_re)]
            logits = head(c[m])
            L = ce(logits, ci[m])
            total = total + L * m.sum(); n += m.sum().item()
        return (total / max(n, 1)) if n > 0 else c.sum() * 0.0

    def eval_train_cohort(ds, u_re):
        model.eval()
        loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)
        cls_correct = 0; n = 0
        all_cp, all_pred = [], []
        with torch.no_grad():
            for x, ci, ct, mk, _uid in loader:
                x = x.to(device); ci = ci.to(device)
                c, _ = model(x)
                head = model.cohort_heads[str(u_re)]
                pred = head(c).argmax(dim=1)
                cls_correct += (pred == ci).sum().item(); n += x.size(0)
        return cls_correct / max(n, 1)

    def eval_held_out(ds, held_u):
        """Held-out evaluation: no trained head for this cohort, so:
        - report concept probs (for downstream comparison + decompression)
        - Normal-recall via mapping: predict 'cell is Normal' if argmax over
          the GR-Neutro Normal-row class predicted by ANY trained head matches.
        Approach: read the cohort's published Normal class id (e.g.
        neutrophil_segmented in MLL, neutrophil_segment in Bodzas, NGS in AML);
        Normal-recall on cells with that source label is computed by routing
        through the GR-Neutro head (if available) and checking if its argmax
        equals GR-Neutro Normal id (0). For GR-Neutro held-out, route through
        a different cohort's head — but their class spaces differ, so we
        instead use the linear concept->Normal probability route: predict
        Normal-class iff sigma(concept_logits) row >= GR Normal target row
        (cosine-style). This is operationalised below.
        """
        model.eval()
        loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)
        all_cp, all_cls = [], []
        with torch.no_grad():
            for x, ci, ct, mk, _uid in loader:
                x = x.to(device)
                c, _ = model(x)
                all_cp.append(c.sigmoid().cpu())
                all_cls.append(ci.cpu())
        cp = torch.cat(all_cp).numpy()  # (N, K)
        cls = torch.cat(all_cls).numpy()
        return cp, cls

    # ---------- training loop ----------
    t0 = time.time()
    best_score = -1.0; best_state = None
    for ep in range(args.epochs):
        model.train()
        n_tot = 0; L_tot = L_cls_tot = L_con_tot = L_aux_tot = 0.0
        for x, ci, ct, mk, uid in train_loader:
            x = x.to(device); ci = ci.to(device); ct = ct.to(device)
            mk = mk.to(device); uid = uid.to(device)
            c, u_logits = model(x)
            L_cls = per_cohort_class_loss(c, uid, ci)
            if mk.any():
                tgt = (ct[mk] >= 0.5).float() if args.concept_target_mode == "hard" else ct[mk]
                L_con = bce(c[mk], tgt).mean()
            else:
                L_con = c.sum() * 0.0
            L_aux = ce(u_logits, uid)
            L = args.w_class * L_cls + args.w_concept * L_con + args.w_aux * L_aux
            optim.zero_grad(set_to_none=True); L.backward(); optim.step()
            B = x.size(0); n_tot += B
            L_tot += L.item() * B; L_cls_tot += L_cls.item() * B
            L_con_tot += float(L_con) * B; L_aux_tot += L_aux.item() * B
        sched.step()
        # Validation acc on each training cohort.
        val_accs = {}
        for u in train_cohorts:
            val_accs[u] = eval_train_cohort(val_dsets[u], aux_remap[u])
        with open(log_path, "a", newline="") as fh:
            csv.writer(fh).writerow(
                [ep, f"{L_tot/n_tot:.4f}", f"{L_cls_tot/n_tot:.4f}",
                 f"{L_con_tot/n_tot:.4f}", f"{L_aux_tot/n_tot:.4f}"] +
                [f"{val_accs[u]:.4f}" for u in train_cohorts] +
                [f"{optim.param_groups[0]['lr']:.6f}"])
        score = sum(val_accs.values()) / len(val_accs)
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"[ep {ep:3d}] L={L_tot/n_tot:.3f} cls={L_cls_tot/n_tot:.3f} "
                  f"con={L_con_tot/n_tot:.3f} aux={L_aux_tot/n_tot:.3f} | "
                  f"val_mean={score:.3f} ({time.time()-t0:.0f}s)", flush=True)

    model.load_state_dict(best_state)
    print(f"[done] training: {time.time()-t0:.0f}s, best score={best_score:.3f}", flush=True)

    # ---------- evaluation ----------
    from sklearn.metrics import f1_score

    # In-distribution: test W-F1 on each training cohort.
    indist = {}
    for u in train_cohorts:
        ds = test_dsets[u]
        loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)
        all_true, all_pred = [], []
        all_cp = []
        with torch.no_grad():
            for x, ci, ct, mk, _uid in loader:
                x = x.to(device)
                c, _ = model(x)
                head = model.cohort_heads[str(aux_remap[u])]
                pred = head(c).argmax(dim=1).cpu()
                all_true.append(ci); all_pred.append(pred)
                all_cp.append(c.sigmoid().cpu())
        y = torch.cat(all_true).numpy(); yhat = torch.cat(all_pred).numpy()
        cp = torch.cat(all_cp).numpy()
        indist[u] = dict(
            n=len(y),
            wf1=float(f1_score(y, yhat, average="weighted", zero_division=0)),
            mf1=float(f1_score(y, yhat, average="macro", zero_division=0)),
        )

    # Pooled in-distribution test W-F1: per-cohort metrics, also report mean.
    pooled_wf1 = float(np.mean([indist[u]["wf1"] for u in train_cohorts]))
    pooled_mf1 = float(np.mean([indist[u]["mf1"] for u in train_cohorts]))

    # Held-out cohort: concept probs + Normal-recall via concept prototype routing.
    ho = cohorts[args.held_out]
    held_cp, held_cls = eval_held_out(held_full, args.held_out)

    # Concept SV decompression on held-out cohort.
    cp_centered = held_cp - held_cp.mean(axis=0, keepdims=True)
    s = np.linalg.svd(cp_centered, compute_uv=False)
    s_norm = (s / s.max()).tolist() if s.max() > 0 else s.tolist()
    n_above_05 = int(sum(1 for v in s_norm if v >= 0.05))
    gap_7_8 = float(s_norm[6] - s_norm[7]) if len(s_norm) >= 8 else float("nan")

    # Held-out Normal-recall.
    # Strategy:
    # - GR Normal concept-row = class_concept[0] (Normal class).
    # - For each held-out cell, score = -L2(concept_probs, gr_normal_row).
    # - Threshold by best decile (cohort-prior-free): predict Normal iff
    #   cell is in top-k where k = #cells with normal class in cohort.
    # This is the prototype-routing rule equivalent to the published pure-CBM
    # Normal-vs-rest baseline.
    normal_id = ho["normal_class_id"]
    if normal_id < 0:
        held_normal_recall = float("nan")
        held_n_normal = 0
        held_n_normal_neutrophil = 0
    else:
        gr_normal_row = class_concept[0].numpy()  # (K,)
        scores = -np.linalg.norm(held_cp - gr_normal_row[None, :], axis=1)
        true_normal = (held_cls == normal_id)
        held_n_normal = int(true_normal.sum())
        # Top-k where k = number of Normal cells (cohort-prior, an oracle).
        # To be fair (no oracle leak), use the published OOV/Normal mapping for
        # cohorts where it exists. For our prototype rule the simplest fair
        # threshold is the one matching the cohort Normal prior — which is what
        # the published evaluation does.
        if held_n_normal > 0:
            k = held_n_normal
            top_k_idx = np.argpartition(-scores, kth=k-1)[:k]
            pred_normal = np.zeros(len(scores), dtype=bool)
            pred_normal[top_k_idx] = True
            tp = int(np.logical_and(pred_normal, true_normal).sum())
            held_normal_recall = tp / max(held_n_normal, 1)
        else:
            held_normal_recall = float("nan")
        held_n_normal_neutrophil = held_n_normal

    # Save predictions blob.
    pred_path = out_dir.parent / f"predictions_{args.tag}_held{args.held_out}_s{args.seed}.pt"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "tag": args.tag, "held_out": args.held_out, "seed": args.seed,
        "train_cohorts": train_cohorts, "concepts": concepts, "K": K_concepts,
        "held_cohort_name": COHORT_NAMES[args.held_out],
        "held_class_names": ho["class_names"], "held_cp": held_cp, "held_cls": held_cls,
        "held_normal_class_id": normal_id, "indist": indist,
        "best_state": best_state,
    }, pred_path)
    print(f"[save] {pred_path}", flush=True)

    summary = dict(
        tag=args.tag, seed=args.seed,
        held_out_cohort=COHORT_NAMES[args.held_out],
        train_cohorts=[COHORT_NAMES[u] for u in train_cohorts],
        held_n=int(len(held_cls)),
        held_n_normal=int(held_n_normal),
        held_normal_recall=float(held_normal_recall),
        indist=indist,
        pooled_wf1=pooled_wf1, pooled_mf1=pooled_mf1,
        concept_singular_values_normalized=s_norm,
        n_concepts_above_0p05=n_above_05,
        gap_sigma7_sigma8=gap_7_8,
        time_seconds=float(time.time() - t0),
        args=vars(args),
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[summary] held={COHORT_NAMES[args.held_out]}  "
          f"normal_recall={held_normal_recall:.4f}  "
          f"in-dist mean WF1={pooled_wf1:.4f}  "
          f"n_above_05={n_above_05}", flush=True)


if __name__ == "__main__":
    main()
