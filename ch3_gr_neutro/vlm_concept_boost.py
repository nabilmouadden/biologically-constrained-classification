#!/usr/bin/env python3
"""VLM-concept-boost test: do HuatuoGPT-Vision per-cell concept scores boost
GR-Neutro classification over the DinoBloom-B backbone alone?

Definitive in-distribution test of the selling claim. Honest result either way.

Conditions (multi-seed, paired-bootstrap CIs):
  1. baseline       : classifier on DinoBloom-B features alone
  2. dino+vlm       : classifier on [DinoBloom-B feats  (+)  HuatuoGPT 11 concept scores]
  3. vlm_alone      : classifier on just the 11 HuatuoGPT concept scores
  (4. dino+stack33  : if >=2 raters cover the corpus -> [DinoBloom (+) stacked R^33]; else skipped)

Labels: dominant_class_idx (7 classes). Metrics: weighted-F1 (W-F1), macro-F1, per-class F1.
Split: data.py stratified_multilabel_split(test_size=0.10, val_size=0.10, seed=SEED),
       varied per seed. Classifier seed also varied.

NO fabrication. Every number from a real run. SLURM only (this script runs on a compute node).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

HERE = Path(__file__).resolve().parent
P3 = HERE / "outputs" / "p3_pilot"
FEATS_NPZ = HERE / "outputs" / "dinobloom_features.npz"
MANIFEST = P3 / "cell_manifest_full_extended.json"
OUT_DIR = HERE / "outputs" / "vlm_concept_boost"

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


def stratified_multilabel_split(labels_np, test_size=0.10, val_size=0.10, seed=42):
    """Verbatim from ch3_gr_neutro/data.py."""
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


def load_huatuo_by_filename():
    """Return {filename: np.array(11,) of concept scores}. NaN-impute missing
    concepts to the per-rater concept mean (computed across covered cells)."""
    recs = {}
    chunks = sorted(P3.glob("huatuogpt_ext_chunk_*.jsonl"))
    if not chunks:
        raise FileNotFoundError(f"no huatuogpt jsonl chunks under {P3}")
    raw = {}
    for ch in chunks:
        with open(ch) as f:
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
                cdict = parsed.get("concepts", {})
                vec = np.full(len(CONCEPTS), np.nan)
                for ki, ck in enumerate(CONCEPTS):
                    item = cdict.get(ck)
                    if item is None:
                        for alias, canon in CONCEPT_ALIASES.items():
                            if canon == ck and alias in cdict:
                                item = cdict[alias]
                                break
                    if isinstance(item, dict):
                        s = item.get("score", np.nan)
                        try:
                            vec[ki] = float(s)
                        except (TypeError, ValueError):
                            pass
                # require >=7/11 concepts valid (matches run_cci_test gate)
                if int(np.sum(~np.isnan(vec))) >= 7:
                    raw[d["filename"]] = vec
    # impute remaining NaNs with column means
    if raw:
        M = np.vstack(list(raw.values()))
        col_mean = np.nanmean(M, axis=0)
        for fn, vec in raw.items():
            nan = np.isnan(vec)
            vec[nan] = col_mean[nan]
            recs[fn] = vec
    return recs


def fit_eval(Xtr, ytr, Xte, yte, seed, n_classes):
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
    clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced",
                             multi_class="multinomial", random_state=seed)
    clf.fit(Xtr_s, ytr)
    pred = clf.predict(Xte_s)
    labels = list(range(n_classes))
    return {
        "pred": pred,
        "wf1": f1_score(yte, pred, average="weighted", labels=labels, zero_division=0),
        "macrof1": f1_score(yte, pred, average="macro", labels=labels, zero_division=0),
        "per_class": f1_score(yte, pred, average=None, labels=labels, zero_division=0).tolist(),
    }


def paired_bootstrap_delta(yte, pred_a, pred_b, class_names, rng, n_boot=N_BOOT):
    """Bootstrap test indices; recompute Delta = metric(b) - metric(a) per resample.
    Returns dict of metric -> (mean_delta, lo, hi) at 95%."""
    yte = np.asarray(yte); pred_a = np.asarray(pred_a); pred_b = np.asarray(pred_b)
    n = len(yte); K = len(class_names); labels = list(range(K))
    d_wf1, d_mf1 = [], []
    d_pc = [[] for _ in range(K)]
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, pa, pb = yte[idx], pred_a[idx], pred_b[idx]
        d_wf1.append(f1_score(yt, pb, average="weighted", labels=labels, zero_division=0)
                     - f1_score(yt, pa, average="weighted", labels=labels, zero_division=0))
        d_mf1.append(f1_score(yt, pb, average="macro", labels=labels, zero_division=0)
                     - f1_score(yt, pa, average="macro", labels=labels, zero_division=0))
        fb = f1_score(yt, pb, average=None, labels=labels, zero_division=0)
        fa = f1_score(yt, pa, average=None, labels=labels, zero_division=0)
        for k in range(K):
            d_pc[k].append(fb[k] - fa[k])

    def ci(arr):
        arr = np.asarray(arr)
        return [float(arr.mean()), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]

    return {
        "d_wf1": ci(d_wf1),
        "d_macrof1": ci(d_mf1),
        "d_per_class": {class_names[k]: ci(d_pc[k]) for k in range(K)},
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[load] manifest + features", flush=True)
    with open(MANIFEST) as f:
        man = json.load(f)
    class_names = man["class_names"]
    cells = man["cells"]
    n_classes = len(class_names)

    npz = np.load(FEATS_NPZ, allow_pickle=True)
    feats = npz["features"]
    paths = npz["paths"]
    feat_by_fn = {Path(str(p)).name: feats[i] for i, p in enumerate(paths)}

    huatuo = load_huatuo_by_filename()
    print(f"[align] manifest={len(cells)} dino_feats={len(feat_by_fn)} "
          f"huatuo_rated={len(huatuo)}", flush=True)

    # keep cells that have BOTH a dino feature AND a huatuo rating (fair paired comparison)
    keep, X_dino, X_vlm, y, labels_oh = [], [], [], [], []
    miss_feat = miss_vlm = 0
    for c in cells:
        fn = c["filename"]
        if fn not in feat_by_fn:
            miss_feat += 1; continue
        if fn not in huatuo:
            miss_vlm += 1; continue
        keep.append(fn)
        X_dino.append(feat_by_fn[fn])
        X_vlm.append(huatuo[fn])
        y.append(c["dominant_class_idx"])
        labels_oh.append(c["label_one_hot"])
    X_dino = np.vstack(X_dino); X_vlm = np.vstack(X_vlm)
    y = np.asarray(y); labels_oh = np.asarray(labels_oh)
    print(f"[align] kept={len(keep)}  miss_feat={miss_feat}  miss_vlm={miss_vlm}", flush=True)
    print(f"[align] class dist: "
          + ", ".join(f"{class_names[k]}={int((y==k).sum())}" for k in range(n_classes)),
          flush=True)

    X_cat = np.hstack([X_dino, X_vlm])
    conditions = {"baseline": X_dino, "dino_plus_vlm": X_cat, "vlm_alone": X_vlm}

    per_seed = {}
    for seed in SEEDS:
        tr_idx, val_idx, te_idx = stratified_multilabel_split(labels_oh, seed=seed)
        # fold val into train (we have no hyperparam search here)
        tr = np.concatenate([tr_idx, val_idx])
        seed_res = {}
        preds = {}
        for name, X in conditions.items():
            r = fit_eval(X[tr], y[tr], X[te_idx], y[te_idx], seed, n_classes)
            preds[name] = r["pred"]
            seed_res[name] = {"wf1": r["wf1"], "macrof1": r["macrof1"],
                              "per_class": r["per_class"]}
        # paired bootstrap: dino_plus_vlm vs baseline
        rng = np.random.default_rng(seed)
        seed_res["delta_vlm_vs_baseline"] = paired_bootstrap_delta(
            y[te_idx], preds["baseline"], preds["dino_plus_vlm"], class_names, rng)
        per_seed[str(seed)] = seed_res
        print(f"[seed {seed}] baseline W-F1={seed_res['baseline']['wf1']:.4f} "
              f"dino+vlm W-F1={seed_res['dino_plus_vlm']['wf1']:.4f} "
              f"vlm_alone W-F1={seed_res['vlm_alone']['wf1']:.4f} "
              f"dWF1={seed_res['delta_vlm_vs_baseline']['d_wf1'][0]:+.4f}", flush=True)

    # ---- aggregate across seeds (mean of point estimates + pooled bootstrap CI) ----
    def agg(metric):
        return {name: {
            "mean": float(np.mean([per_seed[str(s)][name][metric] for s in SEEDS])),
            "std": float(np.std([per_seed[str(s)][name][metric] for s in SEEDS])),
            "per_seed": [per_seed[str(s)][name][metric] for s in SEEDS],
        } for name in conditions}

    def agg_per_class():
        out = {}
        for name in conditions:
            arr = np.array([per_seed[str(s)][name]["per_class"] for s in SEEDS])  # (S,K)
            out[name] = {class_names[k]: {"mean": float(arr[:, k].mean()),
                                          "std": float(arr[:, k].std())}
                         for k in range(n_classes)}
        return out

    # pooled delta CI: average the per-seed bootstrap means, and report
    # the seed-mean delta with across-seed spread + median per-seed CI width.
    d_wf1_means = [per_seed[str(s)]["delta_vlm_vs_baseline"]["d_wf1"][0] for s in SEEDS]
    d_mf1_means = [per_seed[str(s)]["delta_vlm_vs_baseline"]["d_macrof1"][0] for s in SEEDS]
    # seed-level CI on the mean delta (treat per-seed deltas as samples)
    def seed_ci(vals):
        vals = np.asarray(vals)
        return [float(vals.mean()),
                float(vals.mean() - 1.96 * vals.std(ddof=1) / np.sqrt(len(vals))),
                float(vals.mean() + 1.96 * vals.std(ddof=1) / np.sqrt(len(vals)))]

    pooled_delta = {
        "d_wf1_seedmean_ci95": seed_ci(d_wf1_means),
        "d_macrof1_seedmean_ci95": seed_ci(d_mf1_means),
        "per_class_d_seedmean": {},
    }
    for k, cn in enumerate(class_names):
        vals = [per_seed[str(s)]["delta_vlm_vs_baseline"]["d_per_class"][cn][0] for s in SEEDS]
        pooled_delta["per_class_d_seedmean"][cn] = seed_ci(vals)

    results = {
        "experiment": "vlm_concept_boost_indist",
        "backbone": "DinoBloom-B",
        "vlm": "HuatuoGPT-Vision-7B",
        "n_cells_used": len(keep),
        "class_names": class_names,
        "class_dist": {class_names[k]: int((y == k).sum()) for k in range(n_classes)},
        "seeds": SEEDS,
        "n_bootstrap": N_BOOT,
        "conditions": list(conditions),
        "wf1": agg("wf1"),
        "macrof1": agg("macrof1"),
        "per_class_f1": agg_per_class(),
        "delta_vlm_vs_baseline_pooled": pooled_delta,
        "per_seed": per_seed,
        "note": "dino_plus_vlm = [DinoBloom-B 768d (+) HuatuoGPT 11 concept scores]. "
                "vlm_alone = 11 concept scores only. Labels = dominant_class_idx. "
                "Cells restricted to those with BOTH a DinoBloom feature and a HuatuoGPT "
                "rating for a fair paired comparison. LogReg, class_weight=balanced.",
    }
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[done] wrote {OUT_DIR/'results.json'}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
