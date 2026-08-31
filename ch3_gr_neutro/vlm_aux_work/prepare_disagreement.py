"""C4 — Prepare per-cell VLM disagreement vectors as iVAE auxiliary variable.

Reads cached per-cell concept scores for the 5 VLMs (BiomedCLIP, OpenCLIP-b32,
OpenCLIP-l14, PubMedCLIP, PLIP) at
    /gpfs/workdir/mouaddenn/thesis/ch3_gr_neutro/outputs/vlm_grounding/scores_v2/<name>/scores.npz

Computes per-concept disagreement vectors:
    delta_pair(x_c) = | s_A(x_c) - s_B(x_c) |   for each of the 10 unordered pairs
    delta_mean(x_c) = pairwise mean(|s_A - s_B|)
    delta_var(x_c)  = variance across the 5 models per concept

The contribution uses the *primary* pair (BiomedCLIP, OpenCLIP-b32), per task brief:
    delta(x_c) = | s_BiomedCLIP(x_c) - s_OpenCLIP(x_c) |
plus aggregated delta_mean as a robustness check.

Empirically verifies class-independence:
    For each concept k, computes per-class mean delta and the F-statistic of the
    1-way ANOVA across y. Reports the fraction of concepts where p > 0.05
    (i.e. fail-to-reject independence).

Writes:
    delta_primary.npy   (4378, 11) -- absolute disagreement BiomedCLIP vs OpenCLIP-b32
    delta_mean.npy      (4378, 11) -- mean abs pairwise disagreement
    delta_var.npy       (4378, 11) -- per-concept variance across 5 VLMs
    filenames.npy       (4378,)    -- aligned filenames
    class_indep_check.json         -- empirical class-independence report
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy import stats

VLM_DIR = Path("/gpfs/workdir/mouaddenn/thesis/ch3_gr_neutro/outputs/vlm_grounding/scores_v2")
OUT = Path("/gpfs/workdir/mouaddenn/thesis/ch3_gr_neutro/vlm_aux_work")
OUT.mkdir(parents=True, exist_ok=True)

VLMS = ["biomedclip", "openclip_b32", "openclip_l14", "pubmedclip", "plip"]

print("[load] 5 VLM score banks")
banks = {}
for name in VLMS:
    d = np.load(VLM_DIR / name / "scores.npz", allow_pickle=True)
    banks[name] = dict(scores=d["scores"], paths=d["paths"], labels=d["labels"],
                       concepts=list(d["concepts"]), class_names=list(d["class_names"]))
    print(f"  {name}: scores {d['scores'].shape}; paths[0]={d['paths'][0]}")

# Check alignment: same paths and concepts across all VLMs.
ref_paths = banks[VLMS[0]]["paths"]
ref_concepts = banks[VLMS[0]]["concepts"]
ref_labels = banks[VLMS[0]]["labels"]
for name in VLMS[1:]:
    assert list(banks[name]["paths"]) == list(ref_paths), f"path mismatch on {name}"
    assert banks[name]["concepts"] == ref_concepts, f"concept mismatch on {name}"
    assert np.array_equal(banks[name]["labels"], ref_labels), f"label mismatch on {name}"
print(f"[align] {len(ref_paths)} cells aligned across all 5 VLMs; concepts={len(ref_concepts)}")

# Normalise each VLM's scores to [0,1] per concept using empirical min-max,
# so they're on a common scale before we subtract. (Without this,
# BiomedCLIP avg 0.59 vs OpenCLIP avg 0.41 would inflate delta.)
def normalise(s):
    lo = s.min(axis=0, keepdims=True)
    hi = s.max(axis=0, keepdims=True)
    return (s - lo) / np.clip(hi - lo, 1e-6, None)

S = {name: normalise(banks[name]["scores"]) for name in VLMS}
print("[norm] per-concept min-max done")

# Primary pair: BiomedCLIP vs OpenCLIP-b32
delta_primary = np.abs(S["biomedclip"] - S["openclip_b32"]).astype(np.float32)

# Pairwise mean over all 10 unordered pairs
pairs = [(a, b) for i, a in enumerate(VLMS) for b in VLMS[i+1:]]
delta_pairs = np.stack([np.abs(S[a] - S[b]) for (a, b) in pairs], axis=0)  # (10, 4378, 11)
delta_mean = delta_pairs.mean(axis=0).astype(np.float32)

# Per-concept variance across the 5 VLMs (each cell, each concept)
stack5 = np.stack([S[n] for n in VLMS], axis=0)  # (5, 4378, 11)
delta_var = stack5.var(axis=0).astype(np.float32)

print(f"[delta_primary] mean={delta_primary.mean():.4f} std={delta_primary.std():.4f}")
print(f"[delta_mean]    mean={delta_mean.mean():.4f}    std={delta_mean.std():.4f}")
print(f"[delta_var]     mean={delta_var.mean():.4f}     std={delta_var.std():.4f}")

# ---------- Class-independence check ----------
# For each concept k, compute per-class mean disagreement and run a Kruskal-Wallis
# test (non-parametric, robust to non-Normal disagreements). We use the *primary*
# pair delta_primary as the canonical auxiliary signal.
# Each cell has a multi-label vector; assign a "dominant class" = first column == 1,
# defaulting to Normal (column 0). This is a coarse proxy; we also report
# per-class mean using indicator membership.

class_names = banks[VLMS[0]]["class_names"]
K = len(class_names)
labels = ref_labels  # (N, K) int64 multi-label

# Dominant class assignment: argmax across columns; if all zero (shouldn't happen
# for GR-Neutro), assign Normal.
row_sum = labels.sum(axis=1)
dom_class = np.full(labels.shape[0], 0, dtype=int)
nonzero = row_sum > 0
dom_class[nonzero] = labels[nonzero].argmax(axis=1)

indep_report = dict(method="kruskal", aux="delta_primary (BiomedCLIP vs OpenCLIP-b32)",
                    concepts=ref_concepts, class_names=list(class_names),
                    per_concept={})

fraction_indep_at_05 = 0
fraction_indep_at_01 = 0
for j, c in enumerate(ref_concepts):
    per_cls_mean = {}
    per_cls_n = {}
    groups = []
    for k, cls in enumerate(class_names):
        m = labels[:, k] == 1  # cells positive for class k (multi-label)
        if m.sum() > 0:
            vals = delta_primary[m, j]
            per_cls_mean[cls] = float(vals.mean())
            per_cls_n[cls] = int(m.sum())
            groups.append(vals)
    # Test: are these per-class delta distributions all the same?
    h, p = stats.kruskal(*groups)
    fraction_indep_at_05 += int(p > 0.05)
    fraction_indep_at_01 += int(p > 0.01)
    # Also coefficient of variation of per-class means: small CV => class-balanced
    means = np.array(list(per_cls_mean.values()))
    cv = float(means.std() / means.mean()) if means.mean() > 0 else float("nan")
    indep_report["per_concept"][c] = dict(
        per_class_mean=per_cls_mean,
        per_class_n=per_cls_n,
        H_statistic=float(h),
        p_value=float(p),
        coefficient_of_variation_of_means=cv,
        accept_independence_at_alpha_05=bool(p > 0.05),
    )
    print(f"  concept {c[:30]:30s}  p={p:.3g}  CV(means)={cv:.3f}  "
          f"{'INDEP' if p > 0.05 else 'reject indep'}")

indep_report["summary"] = dict(
    fraction_indep_concepts_at_alpha_05=fraction_indep_at_05 / len(ref_concepts),
    fraction_indep_concepts_at_alpha_01=fraction_indep_at_01 / len(ref_concepts),
    n_concepts_indep_at_05=int(fraction_indep_at_05),
    n_concepts_indep_at_01=int(fraction_indep_at_01),
    n_concepts_total=len(ref_concepts),
)
print(f"\n[INDEP SUMMARY] {fraction_indep_at_05}/{len(ref_concepts)} concepts indep@0.05; "
      f"{fraction_indep_at_01}/{len(ref_concepts)} indep@0.01")
print(f"[interpretation] iVAE-compatible auxiliary variable needs delta ⊥ y. "
      f"Fraction of concepts where delta is class-conditional independent of y "
      f"(Kruskal p > 0.05): {fraction_indep_at_05}/{len(ref_concepts)}")

# Save artefacts.
np.save(OUT / "delta_primary.npy", delta_primary)
np.save(OUT / "delta_mean.npy", delta_mean)
np.save(OUT / "delta_var.npy", delta_var)
np.save(OUT / "filenames.npy", ref_paths)
np.save(OUT / "concepts.npy", np.array(ref_concepts))
(OUT / "class_indep_check.json").write_text(json.dumps(indep_report, indent=2))
print(f"[save] {OUT}/delta_primary.npy and others")
print(f"[save] {OUT}/class_indep_check.json")
