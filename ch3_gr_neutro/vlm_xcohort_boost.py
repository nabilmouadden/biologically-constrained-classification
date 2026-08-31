#!/usr/bin/env python3
"""VLM-concept CROSS-COHORT boost test: do HuatuoGPT-Vision per-cell concept
scores improve GR-Neutro -> MLL-23 zero-shot transport (Normal-recall) over the
DinoBloom-B backbone alone?

This is the cross-cohort (out-of-distribution) analogue of the in-distribution
test in vlm_concept_boost.py (which was negative, dWF1=-0.0067). The hypothesis:
under domain shift the class-template breaks down, so per-cell morphology (VLM
concepts) might transport better than the shifted backbone features.

Design:
  TRAIN  : GR-Neutro cells with BOTH a DinoBloom-B feature AND a HuatuoGPT rating.
  TEST   : the rated MLL-23 subset (xcohort_boost early read), cells with BOTH.
  METRIC : Normal-recall = fraction of MLL-23 neutrophil_segmented (Normal) cells
           predicted Normal -- SAME definition as the published 0.473 (pure-CBM)
           and 0.550 (CDA-Maha) baselines, so it is directly comparable.
  CONDITIONS (multi-seed, paired-bootstrap CI over MLL-23 test cells):
     1. baseline      : LogReg on DinoBloom-B features alone
     2. dino_plus_vlm : LogReg on [DinoBloom-B 768d (+) HuatuoGPT 11 concepts]
     3. vlm_alone     : LogReg on the 11 HuatuoGPT concept scores only

  Standardisation is fit on GR-Neutro TRAIN only and applied to MLL-23 (honest
  no-adaptation transport -- the regime where VLM concepts must help). The 11
  VLM concepts are cohort-invariant by construction (identical prompt/schema),
  which is exactly why they are a candidate transport aid.

NO fabrication. Every number from a real run. SLURM/cpu_short only.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

ROOT = Path(os.environ.get(
    "CH3_ROOT", "/gpfs/workdir/mouaddenn/thesis/ch3_gr_neutro"))
P3 = ROOT / "outputs" / "p3_pilot"
GR_FEATS = ROOT / "outputs" / "dinobloom_features.npz"
GR_MANIFEST = P3 / "cell_manifest_full_extended.json"
MLL_FEATS = Path("/gpfs/workdir/mouaddenn/data/mll23/mll23_dinobloom_features.npz")
MLL_MANIFEST = P3 / "cohort_manifest_mll23_xcohort_boost.json"
OUT_DIR = ROOT / "outputs" / "vlm_xcohort_boost"

CLASS_NAMES = ["Normal", "Chromatin", "Dohle", "Hypergranulation",
               "Hypersegmentation", "Hypogranulation", "Hyposegmentation"]
NORMAL_IDX = 0
MLL_NORMAL_SRC = "neutrophil_segmented"  # -> Normal (the recall comparator cells)

CONCEPTS = [
    "nuclear_lobulation_degree", "nuclear_contour_irregularity",
    "nucleus_to_cytoplasm_ratio", "chromatin_condensation_level",
    "chromatin_clumping_pattern", "cytoplasmic_granule_density",
    "granule_coarseness", "cytoplasmic_texture_uniformity",
    "cytoplasm_basophilia_level", "cytoplasmic_inclusion_visibility",
    "cytoplasmic_vacuolization_degree",
]
CONCEPT_ALIASES = {
    "cytoplasmic_basophilia_level": "cytoplasm_basophilia_level",
    "nuclear_to_cytoplasm_ratio": "nucleus_to_cytoplasm_ratio",
    "chromatin_condensation": "chromatin_condensation_level",
    "chromatin_clumping": "chromatin_clumping_pattern",
}
SEEDS = [0, 7, 13, 42, 1337, 2024]
N_BOOT = 5000

# Published cross-cohort MLL-23 Normal-recall references (concept-space CBM):
REF_PURE_CBM = 0.473
REF_CDA = 0.550
REF_ZCA_CDA_SOTA = 0.655  # cross-cohort agent's stain/ZCA-CDA best (cite, not our arm)


def _vec_from_concepts(cdict):
    vec = np.full(len(CONCEPTS), np.nan)
    for ki, ck in enumerate(CONCEPTS):
        item = cdict.get(ck)
        if item is None:
            for alias, canon in CONCEPT_ALIASES.items():
                if canon == ck and alias in cdict:
                    item = cdict[alias]
                    break
        if isinstance(item, dict):
            try:
                vec[ki] = float(item.get("score", np.nan))
            except (TypeError, ValueError):
                pass
    return vec


def load_huatuo_jsonl(paths, col_mean=None):
    """Return ({filename: 11-vec}, col_mean). Requires >=7/11 valid concepts.
    If col_mean is given, impute with it (so MLL-23 uses GR train statistics);
    otherwise compute and return the column mean over covered cells."""
    raw = {}
    for p in paths:
        if not Path(p).exists():
            continue
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                if not d.get("ok", False):
                    continue
                parsed = d.get("parsed", {})
                if parsed.get("_parse_failed"):
                    continue
                vec = _vec_from_concepts(parsed.get("concepts", {}))
                if int(np.sum(~np.isnan(vec))) >= 7:
                    raw[d["filename"]] = vec
    if not raw:
        return {}, col_mean
    M = np.vstack(list(raw.values()))
    cm = col_mean if col_mean is not None else np.nanmean(M, axis=0)
    out = {}
    for fn, vec in raw.items():
        nan = np.isnan(vec)
        vec = vec.copy()
        vec[nan] = cm[nan]
        out[fn] = vec
    return out, cm


def fit_predict(Xtr, ytr, Xte, seed, n_classes):
    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced",
                             multi_class="multinomial", random_state=seed)
    clf.fit(scaler.transform(Xtr), ytr)
    return clf.predict(scaler.transform(Xte))


def normal_recall(pred, mll_sources):
    seg = mll_sources == MLL_NORMAL_SRC
    if seg.sum() == 0:
        return float("nan"), 0
    return float((pred[seg] == NORMAL_IDX).mean()), int(seg.sum())


def paired_bootstrap_recall(seg_mask, pred_a, pred_b, rng, n_boot=N_BOOT):
    """Bootstrap over the segmented (Normal) test cells; Delta = recall(b)-recall(a)."""
    seg_idx = np.where(seg_mask)[0]
    pa = (pred_a[seg_idx] == NORMAL_IDX).astype(float)
    pb = (pred_b[seg_idx] == NORMAL_IDX).astype(float)
    n = len(seg_idx)
    d = []
    for _ in range(n_boot):
        bi = rng.integers(0, n, n)
        d.append(pb[bi].mean() - pa[bi].mean())
    d = np.asarray(d)
    return [float(pb.mean() - pa.mean()), float(np.percentile(d, 2.5)),
            float(np.percentile(d, 97.5))]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_classes = len(CLASS_NAMES)

    # ---- GR-Neutro train ----
    gman = json.load(open(GR_MANIFEST))
    gnpz = np.load(GR_FEATS, allow_pickle=True)
    g_feat_by_fn = {Path(str(p)).name: gnpz["features"][i]
                    for i, p in enumerate(gnpz["paths"])}
    gr_chunks = sorted(P3.glob("huatuogpt_ext_chunk_*.jsonl"))
    gr_vlm, col_mean = load_huatuo_jsonl(gr_chunks)
    print(f"[GR] cells={len(gman['cells'])} feats={len(g_feat_by_fn)} "
          f"vlm_rated={len(gr_vlm)}", flush=True)

    Xtr_d, Xtr_v, ytr = [], [], []
    for c in gman["cells"]:
        fn = c["filename"]
        if fn in g_feat_by_fn and fn in gr_vlm:
            Xtr_d.append(g_feat_by_fn[fn]); Xtr_v.append(gr_vlm[fn])
            ytr.append(c["dominant_class_idx"])
    Xtr_d = np.vstack(Xtr_d); Xtr_v = np.vstack(Xtr_v); ytr = np.asarray(ytr)
    print(f"[GR] train n={len(ytr)} dist="
          + ",".join(f"{CLASS_NAMES[k]}={int((ytr==k).sum())}" for k in range(n_classes)),
          flush=True)

    # ---- MLL-23 test (xcohort_boost rated subset) ----
    mman = json.load(open(MLL_MANIFEST))
    mnpz = np.load(MLL_FEATS, allow_pickle=True)
    m_feat_by_fn = {os.path.basename(str(m)): mnpz["features"][i]
                    for i, m in enumerate(mnpz["members"])}
    m_src_by_fn = {os.path.basename(str(m)): str(mnpz["sources"][i])
                   for i, m in enumerate(mnpz["members"])}
    # impute MLL-23 concept NaNs with the GR-train column means (no target leakage)
    mll_vlm, _ = load_huatuo_jsonl(
        [P3 / "huatuogpt_xcohort_boost_mll23.jsonl"], col_mean=col_mean)
    print(f"[MLL] manifest={len(mman['cells'])} vlm_rated={len(mll_vlm)}", flush=True)

    Xte_d, Xte_v, te_src, te_fn = [], [], [], []
    miss_feat = miss_vlm = 0
    for c in mman["cells"]:
        fn = c["filename"]
        if fn not in m_feat_by_fn:
            miss_feat += 1; continue
        if fn not in mll_vlm:
            miss_vlm += 1; continue
        Xte_d.append(m_feat_by_fn[fn]); Xte_v.append(mll_vlm[fn])
        te_src.append(m_src_by_fn[fn]); te_fn.append(fn)
    Xte_d = np.vstack(Xte_d); Xte_v = np.vstack(Xte_v)
    te_src = np.asarray(te_src)
    print(f"[MLL] test kept={len(te_fn)} miss_feat={miss_feat} miss_vlm={miss_vlm}",
          flush=True)
    seg_mask = te_src == MLL_NORMAL_SRC
    print(f"[MLL] segmented(Normal) test cells (recall denominator) = {int(seg_mask.sum())}",
          flush=True)

    conditions = {
        "baseline": (Xtr_d, Xte_d),
        "dino_plus_vlm": (np.hstack([Xtr_d, Xtr_v]), np.hstack([Xte_d, Xte_v])),
        "vlm_alone": (Xtr_v, Xte_v),
    }

    per_seed = {}
    for seed in SEEDS:
        preds = {}
        rec = {}
        for name, (Xtr, Xte) in conditions.items():
            p = fit_predict(Xtr, ytr, Xte, seed, n_classes)
            preds[name] = p
            r, n_seg = normal_recall(p, te_src)
            rec[name] = {"normal_recall": r, "n_seg": n_seg}
        rng = np.random.default_rng(seed)
        rec["delta_vlm_vs_baseline"] = paired_bootstrap_recall(
            seg_mask, preds["baseline"], preds["dino_plus_vlm"], rng)
        per_seed[str(seed)] = rec
        print(f"[seed {seed}] baseline Nr={rec['baseline']['normal_recall']:.4f} "
              f"dino+vlm Nr={rec['dino_plus_vlm']['normal_recall']:.4f} "
              f"vlm_alone Nr={rec['vlm_alone']['normal_recall']:.4f} "
              f"dNr={rec['delta_vlm_vs_baseline'][0]:+.4f} "
              f"[{rec['delta_vlm_vs_baseline'][1]:+.4f},{rec['delta_vlm_vs_baseline'][2]:+.4f}]",
              flush=True)

    def agg(name):
        vals = np.array([per_seed[str(s)][name]["normal_recall"] for s in SEEDS])
        return {"mean": float(vals.mean()),
                "sd_ddof1": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                "per_seed": vals.tolist()}

    # seed-level CI on the mean delta (per-seed deltas as samples)
    dvals = np.array([per_seed[str(s)]["delta_vlm_vs_baseline"][0] for s in SEEDS])
    delta_seedmean_ci = [
        float(dvals.mean()),
        float(dvals.mean() - 1.96 * dvals.std(ddof=1) / np.sqrt(len(dvals))),
        float(dvals.mean() + 1.96 * dvals.std(ddof=1) / np.sqrt(len(dvals))),
    ]

    results = {
        "experiment": "vlm_concept_boost_crosscohort_mll23",
        "direction": "GR-Neutro (train) -> MLL-23 (zero-shot test)",
        "backbone": "DinoBloom-B",
        "vlm": "HuatuoGPT-Vision-7B",
        "metric": "Normal-recall = frac of MLL-23 neutrophil_segmented predicted Normal "
                  "(same definition as published 0.473/0.550)",
        "n_train_gr": int(len(ytr)),
        "n_test_mll_total": int(len(te_fn)),
        "n_test_mll_segmented_recall_denom": int(seg_mask.sum()),
        "seeds": SEEDS,
        "n_bootstrap": N_BOOT,
        "references": {"pure_cbm": REF_PURE_CBM, "cda_maha": REF_CDA,
                       "zca_cda_sota_crosscohort_agent": REF_ZCA_CDA_SOTA},
        "normal_recall": {name: agg(name) for name in conditions},
        "delta_dino_plus_vlm_vs_baseline_seedmean_ci95": delta_seedmean_ci,
        "per_seed": per_seed,
        "note": "Standardiser fit on GR-Neutro TRAIN only, applied to MLL-23 "
                "(no-adaptation transport). MLL-23 VLM-concept NaNs imputed with "
                "GR-train column means (no target leakage). Cells restricted to "
                "those with BOTH a DinoBloom feature and a HuatuoGPT rating.",
    }
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2))
    print(f"[done] wrote {OUT_DIR/'results.json'}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
