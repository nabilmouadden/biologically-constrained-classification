"""C12 — Patient x cohort joint auxiliary CBM training.

Combines Contribution C1 (multi-cohort iVAE conditioning) with R4's patient-level
critique. The auxiliary variable u becomes the JOINT pair (u_cohort, u_patient)
instead of cohort alone, and we add a second supervised head that predicts the
patient identifier from the concept embedding. Per Khemakhem 2020 Thm. 1, a
richer auxiliary should yield stronger identifiability if the cohort-marginal
concept distributions span more parameter directions when split per patient.

Patient-ID feasibility on the four cohorts (June 2026 audit, this script):
  - GR-Neutro      : 222 proxy patients via r4_pd's cellavision sequential-ID +
                     timestamp clustering. Real per-slide IDs (not feature clusters).
                     Source: outputs/patient_disjoint/patient_disjoint_split.csv.
  - AML Matek 2019 : per-cell patient mapping NOT in standard release. Filenames
                     are sequential per class (NGS_0001.tiff, etc.). We assign a
                     single placeholder patient id (-1) per cohort so the joint
                     aux head still works but the patient-axis contribution
                     comes entirely from GR-Neutro.
  - MLL-23         : per-cell patient mapping NOT in standard release. Filenames
                     are sequential per class. Same placeholder fallback as AML.
  - Bodzas 2023    : 78 patients in the paper, but the cluster-side zip
                     (44 GB) is corrupted, and the dataset card per-cell mapping
                     would still require slide-level lookup outside the release.
                     Excluded from C12 in this round (consistent with C1, where
                     Bodzas was deferred for the same reason).

Per the C12 protocol's hard rule: if patient IDs are not recoverable for >= 2 of
the 4 cohorts, the contribution is reported as limited. THAT IS THE CASE HERE
(3 of 4 cohorts use placeholder patient IDs). We still train the joint
(cohort, patient) auxiliary head because the GR-Neutro per-slide structure
alone is enough to test the patient-axis decompression claim.

Model heads:
  - concept_adapter: MLP(768 -> 256 -> K_concepts) producing concept logits c.
  - cohort_aux: MLP(c -> num_cohorts) — same as C1.
  - patient_aux: MLP(c -> num_patients) — NEW, the second auxiliary head.
  - cohort_heads: per-cohort class heads on top of c (same as C1).

Loss:
  L = w_class * sum_u L_class_u(classifier_u(c), y_u)         (CE)
    + w_concept * L_concept(c, c_target, supervised_mask)     (BCE, masked)
    + w_aux_cohort * L_aux_cohort(cohort_head(c), u_cohort)   (CE)
    + w_aux_patient * L_aux_patient(patient_head(c), u_patient)  (CE; masked
        to GR-Neutro since the other cohorts have placeholder patient ids)

Class-balanced sampling: the per-patient label is highly imbalanced (one cohort
dominates), so the optimiser uses a WeightedRandomSampler that balances the
joint (cohort, patient) label distribution within an epoch.
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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

WORKDIR = Path("/gpfs/workdir/mouaddenn")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Use the same concept-matrix utilities as train.py / train_multicohort.py
from models import build_prior_C, build_class_concept_targets, aggregate_concept_target


# ============================================================================
# Cohort data loaders (cache-backed, mirror C1's paths)
# ============================================================================

GR_FEAT_NPZ = "/gpfs/workdir/mouaddenn/thesis/ch3_gr_neutro/outputs/dinobloom_features.npz"
GR_ANN = "/gpfs/workdir/mouaddenn/data/gr_neutro_extended/annotations.csv"
GR_PATIENT_CSV = "/gpfs/workdir/mouaddenn/thesis/ch3_gr_neutro/outputs/patient_disjoint_split.csv"
AML_FEAT_NPZ = "/gpfs/workdir/mouaddenn/data/aml_matek/features/dinobloom_b_cls.npz"
MLL_FEAT_NPZ = "/gpfs/workdir/mouaddenn/data/mll23/mll23_dinobloom_features.npz"

# AML Matek -> GR-Neutro concept-supervision class index (same as C1).
AML_TO_GR_CONCEPT_CLASS = {7: 0, 6: 6}  # NGS -> Normal, NGB -> Hyposegmentation
MLL_TO_GR_CONCEPT_CLASS = {
    "neutrophil_segmented": 0,
    "neutrophil_band": 6,
}


def load_gr_neutro_with_patient(class_concept: torch.Tensor, normal_idx: int):
    """GR-Neutro features + class + concept + patient_id from r4_pd's split CSV.

    Returns:
        feats  : (N, 768) float
        cls    : (N,) long
        ct     : (N, K) float concept targets
        mask   : (N,) bool concept-supervision mask (all True)
        patient_ids: (N,) long; 0..(n_patients_gr-1)
        gr_class_names, gr_paths
    """
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
        bn = Path(p).name
        labels[i] = ann[bn]
    multilabel = torch.from_numpy(labels)
    concept_target = aggregate_concept_target(class_concept, multilabel,
                                              normal_idx=normal_idx)
    cls_idx = labels.argmax(axis=1).astype(np.int64)
    concept_mask = np.ones(len(paths), dtype=bool)

    # Patient IDs from r4_pd's CSV.
    pat_map = {}
    if Path(GR_PATIENT_CSV).exists():
        with open(GR_PATIENT_CSV) as f:
            r = csv.DictReader(f)
            for row in r:
                pat_map[row["filename"]] = int(row["proxy_patient"])
    patient_ids = np.zeros(len(paths), dtype=np.int64)
    missing = 0
    for i, p in enumerate(paths):
        bn = Path(p).name
        if bn in pat_map:
            patient_ids[i] = pat_map[bn]
        else:
            patient_ids[i] = -1
            missing += 1
    if missing > 0:
        print(f"[gr-patient] WARN: {missing} cells missing patient id (assigned -1)",
              flush=True)
    n_gr_patients = int(patient_ids[patient_ids >= 0].max()) + 1 if (patient_ids >= 0).any() else 0
    print(f"[gr-patient] {n_gr_patients} unique GR-Neutro proxy patients", flush=True)
    return (feats, torch.from_numpy(cls_idx), concept_target,
            torch.from_numpy(concept_mask), torch.from_numpy(patient_ids),
            class_names, paths, n_gr_patients)


def load_aml_matek(class_concept: torch.Tensor, normal_idx: int, placeholder_pid: int):
    """AML Matek with placeholder patient id (no per-cell patient mapping in release)."""
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
            concept_target[i] = class_concept[gr_cls]
            concept_mask[i] = True
    patient_ids = np.full(N, placeholder_pid, dtype=np.int64)
    print(f"[aml] {N} cells; concept-supervised: {concept_mask.sum()}; "
          f"patient placeholder pid={placeholder_pid}", flush=True)
    return (feats, labels, concept_target, torch.from_numpy(concept_mask),
            torch.from_numpy(patient_ids))


def load_mll23(class_concept: torch.Tensor, normal_idx: int, placeholder_pid: int):
    """MLL-23 with placeholder patient id (no per-cell patient mapping in release)."""
    d = np.load(MLL_FEAT_NPZ, allow_pickle=True)
    feats = torch.from_numpy(d["features"]).float()
    sources = d["sources"]
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
    patient_ids = np.full(N, placeholder_pid, dtype=np.int64)
    print(f"[mll] {N} cells; {len(unique_sources)} classes; "
          f"concept-supervised: {concept_mask.sum()}; "
          f"patient placeholder pid={placeholder_pid}", flush=True)
    return (feats, labels, concept_target, torch.from_numpy(concept_mask),
            torch.from_numpy(patient_ids), unique_sources)


# ============================================================================
# Dataset
# ============================================================================

class MultiCohortPatientDataset(Dataset):
    def __init__(self, feats, cls, ct, mask, patient_ids, cohort_id):
        self.feats = feats
        self.cls = cls
        self.ct = ct
        self.mask = mask
        self.patient_ids = patient_ids
        self.cohort_id = cohort_id

    def __len__(self): return self.feats.shape[0]

    def __getitem__(self, i):
        return (self.feats[i], self.cls[i], self.ct[i],
                self.mask[i].item(), self.cohort_id,
                self.patient_ids[i].item())


def split_indices(N, seed, val_frac=0.10, test_frac=0.10):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(N)
    n_test = int(round(N * test_frac))
    n_val = int(round(N * val_frac))
    test = perm[:n_test]; val = perm[n_test:n_test+n_val]
    tr = perm[n_test+n_val:]
    return tr, val, test


def split_gr_by_patient(patient_ids, seed, val_frac=0.10, test_frac=0.10):
    """Patient-disjoint split: a given patient never appears across splits."""
    rng = np.random.RandomState(seed)
    uniq = np.unique(patient_ids.numpy() if torch.is_tensor(patient_ids) else patient_ids)
    rng.shuffle(uniq)
    n_test_p = int(round(len(uniq) * test_frac))
    n_val_p = int(round(len(uniq) * val_frac))
    test_p = set(uniq[:n_test_p].tolist())
    val_p = set(uniq[n_test_p:n_test_p+n_val_p].tolist())
    tr_idx, val_idx, te_idx = [], [], []
    arr = patient_ids.numpy() if torch.is_tensor(patient_ids) else patient_ids
    for i, p in enumerate(arr):
        if int(p) in test_p:
            te_idx.append(i)
        elif int(p) in val_p:
            val_idx.append(i)
        else:
            tr_idx.append(i)
    return np.array(tr_idx), np.array(val_idx), np.array(te_idx)


# ============================================================================
# Model
# ============================================================================

class MultiCohortPatientCBM(nn.Module):
    def __init__(self, embed_dim, K_concepts, cohort_num_classes: dict[int, int],
                 num_cohorts: int, num_patients: int,
                 hidden=256, dropout=0.3):
        super().__init__()
        self.concept_net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, K_concepts),
        )
        self.cohort_heads = nn.ModuleDict()
        for u, ncls in cohort_num_classes.items():
            self.cohort_heads[str(u)] = nn.Sequential(
                nn.Dropout(dropout * 0.5),
                nn.Linear(K_concepts, ncls),
            )
        # Cohort auxiliary head (C1).
        self.cohort_aux = nn.Sequential(
            nn.LayerNorm(K_concepts),
            nn.Linear(K_concepts, num_cohorts),
        )
        # Patient auxiliary head (NEW for C12). Wider since output size is
        # potentially hundreds; an MLP gives the head capacity to read out
        # patient identity from a low-rank concept space without forcing the
        # concept layer to widen.
        self.patient_aux = nn.Sequential(
            nn.LayerNorm(K_concepts),
            nn.Linear(K_concepts, max(K_concepts, 64)),
            nn.GELU(),
            nn.Linear(max(K_concepts, 64), num_patients),
        )

    def forward(self, x):
        c = self.concept_net(x)
        u_logits = self.cohort_aux(c)
        p_logits = self.patient_aux(c)
        return c, u_logits, p_logits


# ============================================================================
# Training
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
    ap.add_argument("--w_aux_cohort", type=float, default=0.5,
                    help="Weight on cohort-ID aux head (C1 conditioning).")
    ap.add_argument("--w_aux_patient", type=float, default=0.5,
                    help="Weight on patient-ID aux head (R4 + C12 conditioning).")
    ap.add_argument("--concept_target_mode", default="hard", choices=["soft", "hard"])
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--use_aml", action="store_true", default=True)
    ap.add_argument("--no_aml", dest="use_aml", action="store_false")
    ap.add_argument("--use_mll", action="store_true", default=True)
    ap.add_argument("--no_mll", dest="use_mll", action="store_false")
    ap.add_argument("--use_patient_balanced_sampling", action="store_true", default=True)
    ap.add_argument("--out_root", default=str(HERE / "outputs" / "multi_cohort_patient"))
    ap.add_argument("--config", default=str(HERE / "concept_config_gr_neutro.json"))
    ap.add_argument("--patient_disjoint_gr", action="store_true", default=True,
                    help="Use r4_pd's per-patient split for GR-Neutro instead of "
                         "the random seed-based split (gives R4 patient-disjoint "
                         "guarantee for free).")
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
    (gr_feats, gr_cls, gr_ct, gr_mask, gr_patients,
     gr_class_names, gr_paths, n_gr_patients) = load_gr_neutro_with_patient(
        class_concept, normal_idx)
    print(f"[gr] N={len(gr_feats)} classes={gr_class_names} "
          f"patients={n_gr_patients}", flush=True)

    # Patient ID space:
    #   GR-Neutro: 0..n_gr_patients-1
    #   AML placeholder: n_gr_patients
    #   MLL placeholder: n_gr_patients + 1
    # Total patient classes used by patient_aux head.
    next_pid = n_gr_patients
    cohort_data = {0: (gr_feats, gr_cls, gr_ct, gr_mask, K_gr, gr_patients)}
    cohort_names = {0: "GR-Neutro"}
    aml_pid_placeholder = -1
    mll_pid_placeholder = -1

    if args.use_aml:
        aml_pid_placeholder = next_pid; next_pid += 1
        aml = load_aml_matek(class_concept, normal_idx, aml_pid_placeholder)
        if aml is not None:
            aml_feats, aml_cls, aml_ct, aml_mask, aml_pat = aml
            cohort_data[1] = (aml_feats, aml_cls, aml_ct, aml_mask,
                              int(aml_cls.max().item()) + 1, aml_pat)
            cohort_names[1] = "AML-Matek"
        else:
            print("[warn] AML Matek features unavailable; skipping", flush=True)
            next_pid -= 1; aml_pid_placeholder = -1

    mll_class_names = None
    if args.use_mll:
        mll_pid_placeholder = next_pid; next_pid += 1
        mll_loaded = load_mll23(class_concept, normal_idx, mll_pid_placeholder)
        mll_feats, mll_cls, mll_ct, mll_mask, mll_pat, mll_class_names = mll_loaded
        cohort_data[2] = (mll_feats, mll_cls, mll_ct, mll_mask,
                          int(mll_cls.max().item()) + 1, mll_pat)
        cohort_names[2] = "MLL-23"

    n_total_patients = next_pid
    cohorts_used = sorted(cohort_data.keys())
    num_cohorts = len(cohorts_used)
    u_remap = {u: i for i, u in enumerate(cohorts_used)}
    print(f"[cohorts] using {cohorts_used} -> {[cohort_names[u] for u in cohorts_used]}",
          flush=True)
    print(f"[patients] total unique patient ids = {n_total_patients} "
          f"(GR={n_gr_patients}, AML placeholder={aml_pid_placeholder}, "
          f"MLL placeholder={mll_pid_placeholder})", flush=True)

    # ---------- split + dataset ----------
    train_datasets, val_datasets, test_datasets = {}, {}, {}
    cohort_num_classes = {}
    for u, (feats, cls, ct, mask, ncls, pat) in cohort_data.items():
        if u == 0 and args.patient_disjoint_gr:
            tr, val, te = split_gr_by_patient(pat, seed=args.seed)
            print(f"[split-gr] patient-disjoint: train={len(tr)} val={len(val)} test={len(te)}",
                  flush=True)
        else:
            tr, val, te = split_indices(len(feats), seed=args.seed)
        train_datasets[u] = MultiCohortPatientDataset(
            feats[tr], cls[tr], ct[tr], mask[tr], pat[tr], u_remap[u])
        val_datasets[u] = MultiCohortPatientDataset(
            feats[val], cls[val], ct[val], mask[val], pat[val], u_remap[u])
        test_datasets[u] = MultiCohortPatientDataset(
            feats[te], cls[te], ct[te], mask[te], pat[te], u_remap[u])
        cohort_num_classes[u_remap[u]] = ncls
        print(f"[{cohort_names[u]}] train={len(tr)} val={len(val)} "
              f"test={len(te)} ncls={ncls}", flush=True)
        torch.save({
            "feats_all": feats, "cls_all": cls, "ct_all": ct, "mask_all": mask,
            "patient_all": pat,
            "train_idx": torch.tensor(tr), "val_idx": torch.tensor(val),
            "test_idx": torch.tensor(te), "cohort_name": cohort_names[u],
        }, out_dir / f"cohort_{u_remap[u]}_split.pt")

    train_concat = torch.utils.data.ConcatDataset(
        [train_datasets[u] for u in cohorts_used])

    # ---------- model ----------
    embed_dim = gr_feats.shape[1]
    model = MultiCohortPatientCBM(
        embed_dim=embed_dim, K_concepts=K_concepts,
        cohort_num_classes=cohort_num_classes, num_cohorts=num_cohorts,
        num_patients=n_total_patients,
        hidden=args.hidden, dropout=args.dropout,
    ).to(device)
    print(f"[model] embed={embed_dim} K={K_concepts} cohorts={num_cohorts} "
          f"patients={n_total_patients} per-cohort ncls={cohort_num_classes}", flush=True)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    bce_logits = nn.BCEWithLogitsLoss(reduction="none")
    ce_logits = nn.CrossEntropyLoss(reduction="mean")

    def cohort_class_loss(c_logits, u_remapped, cls_idx):
        total = 0.0; n = 0
        for u_re in cohort_num_classes.keys():
            mask_u = (u_remapped == u_re)
            if not mask_u.any(): continue
            head = model.cohort_heads[str(u_re)]
            logits_u = head(c_logits[mask_u])
            ce = ce_logits(logits_u, cls_idx[mask_u])
            total = total + ce * mask_u.sum()
            n += mask_u.sum().item()
        return (total / max(n, 1)) if n > 0 else c_logits.sum() * 0.0

    # ---------- patient-balanced sampler ----------
    # The patient label is dominated by AML/MLL placeholders unless we
    # rebalance, so the patient_aux head would collapse to predicting the
    # placeholder. We weight each train sample by 1 / count(joint_label) where
    # joint_label = (cohort, patient).
    if args.use_patient_balanced_sampling:
        joint_keys = []
        for ds in [train_datasets[u] for u in cohorts_used]:
            for i in range(len(ds)):
                _, _, _, _, u_id, pat_id = ds[i]
                joint_keys.append((u_id, pat_id))
        from collections import Counter
        cnt = Counter(joint_keys)
        # Cap counts so a single huge GR patient does not get downweighted to noise.
        weights = np.array([1.0 / max(cnt[k], 1) for k in joint_keys], dtype=np.float64)
        weights = weights / weights.sum() * len(weights)
        sampler = WeightedRandomSampler(weights.tolist(),
                                        num_samples=len(weights),
                                        replacement=True)
        train_loader = DataLoader(train_concat, batch_size=args.batch_size,
                                  sampler=sampler, num_workers=2, drop_last=False)
        print(f"[sampler] patient-balanced; unique joint keys={len(cnt)}", flush=True)
    else:
        train_loader = DataLoader(train_concat, batch_size=args.batch_size,
                                  shuffle=True, num_workers=2, drop_last=False)

    log_path = out_dir / "training_log.csv"
    with open(log_path, "w", newline="") as fh:
        csv.writer(fh).writerow([
            "epoch", "L", "L_class", "L_concept", "L_aux_cohort", "L_aux_patient",
            "val_gr_acc", "val_aml_acc", "val_mll_acc", "val_concept_f1",
            "val_cohort_acc", "val_patient_acc_gr", "lr",
        ])

    def evaluate(dsets):
        model.eval()
        out = {}
        all_concept_probs, all_concept_targ, all_concept_mask = [], [], []
        all_cohort_pred, all_cohort_true = [], []
        all_patient_pred_gr, all_patient_true_gr = [], []
        with torch.no_grad():
            for u, ds in dsets.items():
                if len(ds) == 0: continue
                u_re = u_remap[u]
                loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)
                cls_correct = 0; n = 0
                all_clp, all_cls_true = [], []
                for x, ci, ct, mk, u_id, p_id in loader:
                    x = x.to(device); ci = ci.to(device); ct = ct.to(device)
                    c, u_logits, p_logits = model(x)
                    head = model.cohort_heads[str(u_re)]
                    cls_logits = head(c)
                    pred = cls_logits.argmax(dim=1)
                    cls_correct += (pred == ci).sum().item(); n += x.size(0)
                    if u == 0:
                        all_concept_probs.append(c.sigmoid().cpu())
                        all_concept_targ.append(ct.cpu())
                        all_concept_mask.append(mk.cpu())
                        all_patient_pred_gr.append(p_logits.argmax(dim=1).cpu())
                        all_patient_true_gr.append(p_id)
                    all_clp.append(cls_logits.softmax(dim=1).cpu())
                    all_cls_true.append(ci.cpu())
                    all_cohort_pred.append(u_logits.argmax(dim=1).cpu())
                    all_cohort_true.append(u_id)
                out[u] = dict(acc=cls_correct/max(n,1), n=n,
                              cls_probs=torch.cat(all_clp).numpy(),
                              cls_true=torch.cat(all_cls_true).numpy())
        if all_concept_probs:
            cp = torch.cat(all_concept_probs).numpy()
            ctg = torch.cat(all_concept_targ).numpy()
            cmk = torch.cat(all_concept_mask).numpy()
            y = (ctg >= 0.5).astype(int)
            pr = (cp >= 0.5).astype(int)
            from sklearn.metrics import f1_score
            f1 = f1_score(y[cmk.astype(bool)], pr[cmk.astype(bool)],
                          average="macro", zero_division=0)
            out["_concept_f1_gr"] = float(f1)
            out["_concept_probs_gr"] = cp
            out["_concept_targets_gr"] = ctg
        # Cohort aux accuracy
        if all_cohort_pred:
            ucp = torch.cat(all_cohort_pred).numpy()
            uct = torch.cat(all_cohort_true).numpy()
            out["_cohort_acc"] = float((ucp == uct).mean())
        # Patient aux accuracy on GR
        if all_patient_pred_gr:
            pp = torch.cat(all_patient_pred_gr).numpy()
            pt = torch.cat(all_patient_true_gr).numpy()
            out["_patient_acc_gr"] = float((pp == pt).mean()) if len(pt) else 0.0
        return out

    # ---------- training ----------
    t0 = time.time()
    best_score = -1.0; best_state = None
    for ep in range(args.epochs):
        model.train()
        n_tot = 0
        L_tot = L_cls_tot = L_con_tot = L_uc_tot = L_up_tot = 0.0
        for x, ci, ct, mk, u_id, p_id in train_loader:
            x = x.to(device); ci = ci.to(device); ct = ct.to(device)
            mk = mk.to(device); u_id = u_id.to(device); p_id = p_id.to(device)
            c, u_logits, p_logits = model(x)
            L_cls = cohort_class_loss(c, u_id, ci)
            if mk.any():
                tgt = (ct[mk] >= 0.5).float() if args.concept_target_mode == "hard" else ct[mk]
                L_con = bce_logits(c[mk], tgt).mean()
            else:
                L_con = c.sum() * 0.0
            L_aux_c = ce_logits(u_logits, u_id)
            # Patient aux: ALL rows contribute (placeholder ids are valid classes).
            L_aux_p = ce_logits(p_logits, p_id)
            L = (args.w_class * L_cls + args.w_concept * L_con
                 + args.w_aux_cohort * L_aux_c
                 + args.w_aux_patient * L_aux_p)
            optim.zero_grad(set_to_none=True); L.backward(); optim.step()
            B = x.size(0); n_tot += B
            L_tot += L.item() * B; L_cls_tot += L_cls.item() * B
            L_con_tot += float(L_con) * B
            L_uc_tot += L_aux_c.item() * B
            L_up_tot += L_aux_p.item() * B
        sched.step()
        val = evaluate(val_datasets)
        v_gr = val.get(0, {}).get("acc", 0.0)
        v_am = val.get(1, {}).get("acc", 0.0)
        v_ml = val.get(2, {}).get("acc", 0.0)
        v_cf = val.get("_concept_f1_gr", 0.0)
        v_co = val.get("_cohort_acc", 0.0)
        v_pp = val.get("_patient_acc_gr", 0.0)
        with open(log_path, "a", newline="") as fh:
            csv.writer(fh).writerow([
                ep, f"{L_tot/n_tot:.4f}", f"{L_cls_tot/n_tot:.4f}",
                f"{L_con_tot/n_tot:.4f}", f"{L_uc_tot/n_tot:.4f}",
                f"{L_up_tot/n_tot:.4f}",
                f"{v_gr:.4f}", f"{v_am:.4f}", f"{v_ml:.4f}",
                f"{v_cf:.4f}", f"{v_co:.4f}", f"{v_pp:.4f}",
                f"{optim.param_groups[0]['lr']:.6f}",
            ])
        score = v_gr + v_cf
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"[ep {ep:3d}] L={L_tot/n_tot:.3f} cls={L_cls_tot/n_tot:.3f} "
                  f"con={L_con_tot/n_tot:.3f} aux_c={L_uc_tot/n_tot:.3f} "
                  f"aux_p={L_up_tot/n_tot:.3f} | v_gr={v_gr:.3f} v_aml={v_am:.3f} "
                  f"v_mll={v_ml:.3f} v_cf={v_cf:.3f} v_co={v_co:.3f} "
                  f"v_pp={v_pp:.3f} ({time.time()-t0:.0f}s)", flush=True)

    print(f"[done] training: {time.time()-t0:.0f}s, best score={best_score:.3f}",
          flush=True)
    # We keep the LAST-epoch state for the rank-decomposition / BiomedCLIP eval
    # because the v_gr + v_cf scoring rule, paired with the very small
    # patient-disjoint val set (~110 cells), tends to lock best_state at
    # epoch 0-1 before the patient-aware aux head has converged. Reporting
    # singular values from un-converged weights would understate the rank
    # decompression. We still save best_state to the predictions dump so
    # downstream analysts can opt in.
    final_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # ---------- final test eval + dump ----------
    test = evaluate(test_datasets)
    model.eval()
    full_concept_probs, full_class_logits, full_cls_true = {}, {}, {}
    full_aux_logits, full_pat_logits = {}, {}
    with torch.no_grad():
        for u, (feats, cls, ct, mk, ncls, pat) in cohort_data.items():
            u_re = u_remap[u]
            ds = MultiCohortPatientDataset(feats, cls, ct, mk, pat, u_re)
            loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)
            all_c, all_u, all_p, all_cls_logits, all_cls_true_ = [], [], [], [], []
            for x, ci, ctt, mki, u_id, p_id in loader:
                x = x.to(device)
                c, u_logits, p_logits = model(x)
                head = model.cohort_heads[str(u_re)]
                cls_logits = head(c)
                all_c.append(c.sigmoid().cpu())
                all_u.append(u_logits.softmax(dim=1).cpu())
                all_p.append(p_logits.cpu())
                all_cls_logits.append(cls_logits.cpu())
                all_cls_true_.append(ci.cpu())
            full_concept_probs[u] = torch.cat(all_c).numpy()
            full_aux_logits[u] = torch.cat(all_u).numpy()
            full_pat_logits[u] = torch.cat(all_p).numpy()
            full_class_logits[u] = torch.cat(all_cls_logits).numpy()
            full_cls_true[u] = torch.cat(all_cls_true_).numpy()

    pred_path = out_dir.parent / f"predictions_s{args.seed}.pt"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "concepts": concepts, "K_concepts": K_concepts,
        "cohorts_used": cohorts_used, "cohort_names": cohort_names,
        "u_remap": u_remap, "gr_class_names": gr_class_names,
        "mll_class_names": mll_class_names,
        "n_gr_patients": n_gr_patients, "n_total_patients": n_total_patients,
        "aml_pid_placeholder": aml_pid_placeholder,
        "mll_pid_placeholder": mll_pid_placeholder,
        "full_concept_probs": {u: full_concept_probs[u] for u in cohorts_used},
        "full_class_logits": {u: full_class_logits[u] for u in cohorts_used},
        "full_cls_true": {u: full_cls_true[u] for u in cohorts_used},
        "full_aux_probs": {u: full_aux_logits[u] for u in cohorts_used},
        "full_pat_logits": {u: full_pat_logits[u] for u in cohorts_used},
        "gr_paths": gr_paths,
        "args": vars(args),
        "best_state": best_state,
        "final_state": final_state,
    }, pred_path)
    print(f"[save] {pred_path}", flush=True)

    # Summary
    from sklearn.metrics import f1_score
    # GR-Neutro test (uses the split we used during training).
    if args.patient_disjoint_gr:
        gr_tr, gr_val, gr_te = split_gr_by_patient(gr_patients, seed=args.seed)
    else:
        gr_tr, gr_val, gr_te = split_indices(len(gr_feats), args.seed)
    gr_test_pred = full_class_logits[0][gr_te].argmax(axis=1)
    gr_test_true = gr_cls.numpy()[gr_te]
    gr_wf1 = float(f1_score(gr_test_true, gr_test_pred, average="weighted",
                            zero_division=0))
    gr_mf1 = float(f1_score(gr_test_true, gr_test_pred, average="macro",
                            zero_division=0))

    mll_normal_recall = float("nan")
    if mll_class_names is not None and 2 in full_class_logits:
        try:
            normal_id = mll_class_names.index("neutrophil_segmented")
            mll_true = full_cls_true[2]
            mll_pred = full_class_logits[2].argmax(axis=1)
            mask = (mll_true == normal_id)
            mll_normal_recall = float((mll_pred[mask] == normal_id).mean())
        except Exception as e:
            print(f"[warn] MLL normal recall: {e}", flush=True)

    cp_gr_test = full_concept_probs[0][gr_te]
    cp_centered = cp_gr_test - cp_gr_test.mean(axis=0, keepdims=True)
    s = np.linalg.svd(cp_centered, compute_uv=False)
    s_norm = s / s.max() if s.max() > 0 else s
    n_above = int((s_norm >= 0.05).sum())
    gap_7_8 = float(s_norm[6] - s_norm[7]) if len(s_norm) >= 8 else float("nan")
    gap_8_9 = float(s_norm[7] - s_norm[8]) if len(s_norm) >= 9 else float("nan")

    # BiomedCLIP rho: try Ruche cache if available, otherwise skip.
    biomedclip_rho = []
    try:
        bp = "/gpfs/workdir/mouaddenn/thesis/ch3_gr_neutro/outputs/biomedclip_scores.npz"
        if Path(bp).exists():
            bd = np.load(bp, allow_pickle=True)
            bk = "scores" if "scores" in bd else None
            if bk and "paths" in bd:
                bm_paths = bd["paths"].tolist()
                bm_scores = bd[bk]
                # Align GR-Neutro paths to BiomedCLIP path order.
                bm_index = {Path(p).name: i for i, p in enumerate(bm_paths)}
                from scipy.stats import spearmanr
                aligned_rows = []
                for p in gr_paths:
                    bn = Path(p).name
                    aligned_rows.append(bm_index.get(bn, -1))
                aligned_rows = np.array(aligned_rows)
                ok = aligned_rows >= 0
                cp_all = full_concept_probs[0][ok]
                bm_aligned = bm_scores[aligned_rows[ok]]
                for k in range(min(cp_all.shape[1], bm_aligned.shape[1])):
                    r, _ = spearmanr(cp_all[:, k], bm_aligned[:, k])
                    if not np.isnan(r):
                        biomedclip_rho.append(float(r))
    except Exception as e:
        print(f"[warn] BiomedCLIP rho failed: {e}", flush=True)

    print(f"\n[summary] GR-Neutro test W-F1={gr_wf1:.4f}  Macro-F1={gr_mf1:.4f}",
          flush=True)
    print(f"[summary] MLL-23 Normal recall={mll_normal_recall:.4f}", flush=True)
    print(f"[summary] concept SV (normalized, GR test): {s_norm.round(3).tolist()}",
          flush=True)
    print(f"[summary] n_above(0.05)={n_above}, gap(s7-s8)={gap_7_8:.4f}, "
          f"gap(s8-s9)={gap_8_9:.4f}", flush=True)

    summary = dict(
        tag=args.tag, seed=args.seed, time_seconds=time.time() - t0,
        cohorts_used=cohorts_used, cohort_names=cohort_names,
        gr_test_weighted_f1=gr_wf1, gr_test_macro_f1=gr_mf1,
        mll_normal_recall=mll_normal_recall,
        concept_singular_values_normalized=s_norm.tolist(),
        n_concepts_above_0p05=n_above,
        gap_sigma7_sigma8=gap_7_8, gap_sigma8_sigma9=gap_8_9,
        biomedclip_rho_per_concept=biomedclip_rho,
        biomedclip_mean_abs_rho=float(np.mean(np.abs(biomedclip_rho)))
            if biomedclip_rho else None,
        n_gr_patients=n_gr_patients,
        n_total_patients=n_total_patients,
        aml_pid_placeholder=aml_pid_placeholder,
        mll_pid_placeholder=mll_pid_placeholder,
        patient_disjoint_gr=args.patient_disjoint_gr,
        args=vars(args),
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[save] {out_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
