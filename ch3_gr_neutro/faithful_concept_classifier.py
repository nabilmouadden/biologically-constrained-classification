#!/usr/bin/env python3
"""FAITHFUL-concept classifier + the four payoff conditions.

We now have per-cell concepts by DIRECT MEASUREMENT (morphometry), not VLM
hallucination. This re-asks the questions that FAILED with unfaithful VLM
concepts (vlm_concept_boost: dino+vlm HURT the backbone, dWF1 = -0.0067).

Conditions (GR-Neutro 7-class; data.py stratified_multilabel_split; multi-seed; paired bootstrap):
  1. morpho_alone   : LogReg on the 11 measured concept values (fully transparent classifier)
                      also report a 7-reliable-concepts-only variant.
  2. dino+morpho vs dino : do FAITHFUL concepts BOOST where unfaithful VLM ones HURT?
                      paired-bootstrap dWF1 + per-class (esp. lobulation/granule classes).
  3. faithful CBM   : route classification THROUGH the concept layer
                      (x -> concept-head -> predicted concepts -> class-head).
                      report accuracy + interpretability (read off why each cell got its class).
  4. concept intervention : correct a concept to its true measured value on the
                      concept-bottleneck; does the prediction move sensibly?
                      (force high lobe count -> push toward Hypersegmentation; etc.)

Labels: dominant_class_idx (7 classes). Metrics: weighted-F1 (W-F1), macro-F1, per-class F1.
Mirrors vlm_concept_boost.py exactly (split, seeds, LogReg, paired bootstrap) for a
fair FAITHFUL-vs-UNFAITHFUL contrast.

NO fabrication. SLURM only (cached features + a CSV; cpu_short).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, r2_score

HERE = Path(__file__).resolve().parent
OUT_ROOT = HERE / "outputs"
FEATS_NPZ = OUT_ROOT / "dinobloom_features.npz"
MANIFEST = OUT_ROOT / "p3_pilot" / "cell_manifest_full_extended.json"
MORPHO_CSV = OUT_ROOT / "morphometry_concepts" / "morphometry_concepts.csv"
OUT_DIR = OUT_ROOT / "faithful_concept_classifier"

# 11 measured concept columns in the morphometry CSV (the computed concept VALUES,
# NOT the one-hot class columns).
CONCEPTS = [
    "nuclear_lobulation_degree", "nuclear_contour_irregularity",
    "nucleus_to_cytoplasm_ratio", "chromatin_condensation_level",
    "chromatin_clumping_pattern", "cytoplasmic_granule_density",
    "granule_coarseness", "cytoplasmic_texture_uniformity",
    "cytoplasm_basophilia_level", "cytoplasmic_inclusion_visibility",
    "cytoplasmic_vacuolization_degree",
]
# 7 reliable concepts (AUC>=0.60 & varying) per outputs/morphometry_concepts/validation.md
RELIABLE = [
    "nuclear_lobulation_degree", "nuclear_contour_irregularity",
    "chromatin_clumping_pattern", "cytoplasmic_granule_density",
    "cytoplasmic_texture_uniformity", "cytoplasm_basophilia_level",
    "cytoplasmic_vacuolization_degree",
]
# raw measured columns useful for intervention sanity (interpretable units)
RAW_EXTRA = ["lobe_count", "nc_ratio_raw", "granule_count"]

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


def load_morpho_by_filename():
    """Return {filename: dict(col->float)} of measured concepts + raw extras.
    Only rows with seg_ok==1 are kept (segmentation succeeded)."""
    recs = {}
    with open(MORPHO_CSV) as f:
        r = csv.DictReader(f)
        for row in r:
            if str(row.get("seg_ok", "1")).strip() not in ("1", "1.0", "True", "true"):
                continue
            d = {}
            ok = True
            for c in CONCEPTS + RAW_EXTRA:
                v = row.get(c, "")
                try:
                    d[c] = float(v)
                except (TypeError, ValueError):
                    ok = False
                    break
            if ok:
                recs[row["filename"]] = d
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
        "acc": accuracy_score(yte, pred),
        "per_class": f1_score(yte, pred, average=None, labels=labels, zero_division=0).tolist(),
        "scaler": scaler, "clf": clf,
    }


def paired_bootstrap_delta(yte, pred_a, pred_b, class_names, rng, n_boot=N_BOOT):
    """Delta = metric(b) - metric(a) per resample. 95% CIs."""
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


def seed_ci(vals):
    vals = np.asarray(vals, dtype=float)
    m = vals.mean()
    if len(vals) > 1:
        half = 1.96 * vals.std(ddof=1) / np.sqrt(len(vals))
    else:
        half = 0.0
    return [float(m), float(m - half), float(m + half)]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[load] manifest + features + morphometry csv", flush=True)
    with open(MANIFEST) as f:
        man = json.load(f)
    class_names = man["class_names"]
    cells = man["cells"]
    n_classes = len(class_names)

    npz = np.load(FEATS_NPZ, allow_pickle=True)
    feats = npz["features"]
    paths = npz["paths"]
    feat_by_fn = {Path(str(p)).name: feats[i] for i, p in enumerate(paths)}

    morpho = load_morpho_by_filename()
    print(f"[align] manifest={len(cells)} dino_feats={len(feat_by_fn)} "
          f"morpho_measured={len(morpho)}", flush=True)

    # keep cells with BOTH a dino feature AND a morphometry measurement (fair paired comparison)
    keep, X_dino, M_all, M_rel, RAW, y, labels_oh = [], [], [], [], [], [], []
    miss_feat = miss_morpho = 0
    rel_idx = [CONCEPTS.index(c) for c in RELIABLE]
    for c in cells:
        fn = c["filename"]
        if fn not in feat_by_fn:
            miss_feat += 1; continue
        if fn not in morpho:
            miss_morpho += 1; continue
        keep.append(fn)
        X_dino.append(feat_by_fn[fn])
        mvec = np.array([morpho[fn][k] for k in CONCEPTS], dtype=float)
        M_all.append(mvec)
        M_rel.append(mvec[rel_idx])
        RAW.append(np.array([morpho[fn][k] for k in RAW_EXTRA], dtype=float))
        y.append(c["dominant_class_idx"])
        labels_oh.append(c["label_one_hot"])
    X_dino = np.vstack(X_dino); M_all = np.vstack(M_all); M_rel = np.vstack(M_rel)
    RAW = np.vstack(RAW)
    y = np.asarray(y); labels_oh = np.asarray(labels_oh)
    print(f"[align] kept={len(keep)}  miss_feat={miss_feat}  miss_morpho={miss_morpho}", flush=True)
    print("[align] class dist: "
          + ", ".join(f"{class_names[k]}={int((y==k).sum())}" for k in range(n_classes)),
          flush=True)

    X_cat = np.hstack([X_dino, M_all])  # dino + faithful concepts
    # conditions evaluated head-to-head with one LogReg each
    conditions = {
        "baseline_dino": X_dino,
        "dino_plus_morpho": X_cat,
        "morpho_alone": M_all,
        "morpho_alone_reliable7": M_rel,
    }

    per_seed = {}
    cbm_per_seed = {}
    interv_per_seed = {}
    for seed in SEEDS:
        tr_idx, val_idx, te_idx = stratified_multilabel_split(labels_oh, seed=seed)
        tr = np.concatenate([tr_idx, val_idx])  # no hyperparam search
        seed_res = {}
        preds = {}
        fitted = {}
        for name, X in conditions.items():
            r = fit_eval(X[tr], y[tr], X[te_idx], y[te_idx], seed, n_classes)
            preds[name] = r["pred"]
            fitted[name] = r
            seed_res[name] = {"wf1": r["wf1"], "macrof1": r["macrof1"],
                              "acc": r["acc"], "per_class": r["per_class"]}
        # cond 2: paired bootstrap dino+morpho vs dino
        rng = np.random.default_rng(seed)
        seed_res["delta_morpho_vs_baseline"] = paired_bootstrap_delta(
            y[te_idx], preds["baseline_dino"], preds["dino_plus_morpho"], class_names, rng)
        per_seed[str(seed)] = seed_res

        # ---- cond 3: FAITHFUL CBM (route through concept bottleneck) ----
        # concept head: x_dino -> predict each measured concept (ridge regression).
        # class head:   predicted concepts -> class (LogReg). Decisions trace to concepts.
        from sklearn.linear_model import Ridge
        cscaler = StandardScaler().fit(X_dino[tr])
        Xtr_d = cscaler.transform(X_dino[tr]); Xte_d = cscaler.transform(X_dino[te_idx])
        # standardise concept targets for stable ridge, keep inverse for intervention
        tscaler = StandardScaler().fit(M_all[tr])
        Ctr = tscaler.transform(M_all[tr])
        concept_heads = []
        chat_tr = np.zeros_like(Ctr)
        chat_te = np.zeros((len(te_idx), len(CONCEPTS)))
        concept_r2 = {}
        for j, cn in enumerate(CONCEPTS):
            reg = Ridge(alpha=10.0, random_state=seed).fit(Xtr_d, Ctr[:, j])
            concept_heads.append(reg)
            chat_tr[:, j] = reg.predict(Xtr_d)
            pj = reg.predict(Xte_d)
            chat_te[:, j] = pj
            concept_r2[cn] = float(r2_score(
                tscaler.transform(M_all[te_idx])[:, j], pj))
        # class head on PREDICTED concepts (the bottleneck)
        cls_head = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced",
                                      multi_class="multinomial", random_state=seed)
        cls_head.fit(chat_tr, y[tr])
        cbm_pred = cls_head.predict(chat_te)
        labels = list(range(n_classes))
        cbm_res = {
            "wf1": f1_score(y[te_idx], cbm_pred, average="weighted", labels=labels, zero_division=0),
            "macrof1": f1_score(y[te_idx], cbm_pred, average="macro", labels=labels, zero_division=0),
            "acc": accuracy_score(y[te_idx], cbm_pred),
            "per_class": f1_score(y[te_idx], cbm_pred, average=None, labels=labels, zero_division=0).tolist(),
            "concept_r2": concept_r2,
            "mean_concept_r2": float(np.mean(list(concept_r2.values()))),
        }
        # paired bootstrap: CBM vs baseline_dino
        rng2 = np.random.default_rng(seed + 1)
        cbm_res["delta_cbm_vs_baseline"] = paired_bootstrap_delta(
            y[te_idx], preds["baseline_dino"], cbm_pred, class_names, rng2)
        cbm_per_seed[str(seed)] = cbm_res

        # ---- cond 4: concept intervention sanity ----
        # On the CBM bottleneck, replace the PREDICTED concept value with the TRUE
        # MEASURED value (z-scored same as training targets), one concept at a time,
        # and measure how class probabilities shift. Sanity directions:
        #   nuclear_lobulation_degree HIGH -> P(Hypersegmentation) up, P(Hyposegmentation) down
        #   cytoplasmic_granule_density HIGH -> P(Hypergranulation) up, P(Hypogranulation) down
        Ctrue_te = tscaler.transform(M_all[te_idx])  # true measured, z-scored
        base_proba = cls_head.predict_proba(chat_te)
        ci_hyper = class_names.index("Hypersegmentation")
        ci_hypo = class_names.index("Hyposegmentation")
        ci_hgran = class_names.index("Hypergranulation")
        ci_lgran = class_names.index("Hypogranulation")

        def intervene_set(concept_name, target_z):
            """Set one concept on the bottleneck to target_z (scalar, z-units) for ALL
            test cells; return mean prob shift on each class vs base."""
            j = CONCEPTS.index(concept_name)
            Cmod = chat_te.copy()
            Cmod[:, j] = target_z
            p = cls_head.predict_proba(Cmod)
            return (p - base_proba).mean(axis=0)

        def intervene_true(concept_name):
            """Replace predicted with TRUE measured value for one concept; correlate
            the per-cell true value with the resulting prob change on a target class."""
            j = CONCEPTS.index(concept_name)
            Cmod = chat_te.copy()
            Cmod[:, j] = Ctrue_te[:, j]
            p = cls_head.predict_proba(Cmod)
            return p - base_proba, Cmod[:, j]

        # high = +2 z (well above mean), low = -2 z
        lob_hi = intervene_set("nuclear_lobulation_degree", 2.0)
        lob_lo = intervene_set("nuclear_lobulation_degree", -2.0)
        gran_hi = intervene_set("cytoplasmic_granule_density", 2.0)
        gran_lo = intervene_set("cytoplasmic_granule_density", -2.0)
        # monotonicity: correlate true lobulation value with dP(Hyperseg) when corrected
        dP_lob, lobval = intervene_true("nuclear_lobulation_degree")
        dP_gran, granval = intervene_true("cytoplasmic_granule_density")
        corr_lob_hyper = float(np.corrcoef(lobval, dP_lob[:, ci_hyper])[0, 1])
        corr_gran_hgran = float(np.corrcoef(granval, dP_gran[:, ci_hgran])[0, 1])

        interv_per_seed[str(seed)] = {
            "force_high_lobulation": {
                "dP_Hypersegmentation": float(lob_hi[ci_hyper]),
                "dP_Hyposegmentation": float(lob_hi[ci_hypo]),
            },
            "force_low_lobulation": {
                "dP_Hypersegmentation": float(lob_lo[ci_hyper]),
                "dP_Hyposegmentation": float(lob_lo[ci_hypo]),
            },
            "force_high_granule_density": {
                "dP_Hypergranulation": float(gran_hi[ci_hgran]),
                "dP_Hypogranulation": float(gran_hi[ci_lgran]),
            },
            "force_low_granule_density": {
                "dP_Hypergranulation": float(gran_lo[ci_hgran]),
                "dP_Hypogranulation": float(gran_lo[ci_lgran]),
            },
            "corr_trueLobulation_vs_dP_Hyperseg": corr_lob_hyper,
            "corr_trueGranuleDensity_vs_dP_Hypergran": corr_gran_hgran,
        }

        print(f"[seed {seed}] dino W-F1={seed_res['baseline_dino']['wf1']:.4f} "
              f"dino+morpho={seed_res['dino_plus_morpho']['wf1']:.4f} "
              f"morpho_alone={seed_res['morpho_alone']['wf1']:.4f} "
              f"rel7={seed_res['morpho_alone_reliable7']['wf1']:.4f} "
              f"CBM={cbm_res['wf1']:.4f} "
              f"dWF1(morpho)={seed_res['delta_morpho_vs_baseline']['d_wf1'][0]:+.4f}", flush=True)

    # ---- aggregate ----
    def agg(metric):
        return {name: {
            "mean": float(np.mean([per_seed[str(s)][name][metric] for s in SEEDS])),
            "std": float(np.std([per_seed[str(s)][name][metric] for s in SEEDS])),
            "per_seed": [per_seed[str(s)][name][metric] for s in SEEDS],
        } for name in conditions}

    def agg_per_class():
        out = {}
        for name in conditions:
            arr = np.array([per_seed[str(s)][name]["per_class"] for s in SEEDS])
            out[name] = {class_names[k]: {"mean": float(arr[:, k].mean()),
                                          "std": float(arr[:, k].std())}
                         for k in range(n_classes)}
        return out

    # cond2 pooled delta
    d_wf1_means = [per_seed[str(s)]["delta_morpho_vs_baseline"]["d_wf1"][0] for s in SEEDS]
    d_mf1_means = [per_seed[str(s)]["delta_morpho_vs_baseline"]["d_macrof1"][0] for s in SEEDS]
    pooled_delta = {
        "d_wf1_seedmean_ci95": seed_ci(d_wf1_means),
        "d_macrof1_seedmean_ci95": seed_ci(d_mf1_means),
        "per_class_d_seedmean": {},
    }
    for cn in class_names:
        vals = [per_seed[str(s)]["delta_morpho_vs_baseline"]["d_per_class"][cn][0] for s in SEEDS]
        pooled_delta["per_class_d_seedmean"][cn] = seed_ci(vals)

    # cbm aggregate
    cbm_agg = {
        "wf1": {"mean": float(np.mean([cbm_per_seed[str(s)]["wf1"] for s in SEEDS])),
                "std": float(np.std([cbm_per_seed[str(s)]["wf1"] for s in SEEDS])),
                "per_seed": [cbm_per_seed[str(s)]["wf1"] for s in SEEDS]},
        "macrof1": {"mean": float(np.mean([cbm_per_seed[str(s)]["macrof1"] for s in SEEDS])),
                    "std": float(np.std([cbm_per_seed[str(s)]["macrof1"] for s in SEEDS]))},
        "acc": {"mean": float(np.mean([cbm_per_seed[str(s)]["acc"] for s in SEEDS])),
                "std": float(np.std([cbm_per_seed[str(s)]["acc"] for s in SEEDS]))},
        "mean_concept_r2": {"mean": float(np.mean([cbm_per_seed[str(s)]["mean_concept_r2"] for s in SEEDS])),
                            "std": float(np.std([cbm_per_seed[str(s)]["mean_concept_r2"] for s in SEEDS]))},
        "concept_r2_seedmean": {cn: float(np.mean([cbm_per_seed[str(s)]["concept_r2"][cn] for s in SEEDS]))
                                for cn in CONCEPTS},
        "per_class_f1_seedmean": {class_names[k]: float(np.mean(
            [cbm_per_seed[str(s)]["per_class"][k] for s in SEEDS])) for k in range(n_classes)},
        "delta_cbm_vs_baseline_seedmean": seed_ci(
            [cbm_per_seed[str(s)]["delta_cbm_vs_baseline"]["d_wf1"][0] for s in SEEDS]),
    }

    # intervention aggregate (mean over seeds)
    def imean(path):
        vals = []
        for s in SEEDS:
            d = interv_per_seed[str(s)]
            for k in path:
                d = d[k]
            vals.append(d)
        return seed_ci(vals)

    interv_agg = {
        "force_high_lobulation": {
            "dP_Hypersegmentation": imean(["force_high_lobulation", "dP_Hypersegmentation"]),
            "dP_Hyposegmentation": imean(["force_high_lobulation", "dP_Hyposegmentation"]),
        },
        "force_low_lobulation": {
            "dP_Hypersegmentation": imean(["force_low_lobulation", "dP_Hypersegmentation"]),
            "dP_Hyposegmentation": imean(["force_low_lobulation", "dP_Hyposegmentation"]),
        },
        "force_high_granule_density": {
            "dP_Hypergranulation": imean(["force_high_granule_density", "dP_Hypergranulation"]),
            "dP_Hypogranulation": imean(["force_high_granule_density", "dP_Hypogranulation"]),
        },
        "force_low_granule_density": {
            "dP_Hypergranulation": imean(["force_low_granule_density", "dP_Hypergranulation"]),
            "dP_Hypogranulation": imean(["force_low_granule_density", "dP_Hypogranulation"]),
        },
        "corr_trueLobulation_vs_dP_Hyperseg": imean(["corr_trueLobulation_vs_dP_Hyperseg"]),
        "corr_trueGranuleDensity_vs_dP_Hypergran": imean(["corr_trueGranuleDensity_vs_dP_Hypergran"]),
    }

    results = {
        "experiment": "faithful_concept_classifier",
        "backbone": "DinoBloom-B",
        "concepts": "morphometry (direct measurement), 11 computed",
        "reliable_concepts": RELIABLE,
        "n_cells_used": len(keep),
        "class_names": class_names,
        "class_dist": {class_names[k]: int((y == k).sum()) for k in range(n_classes)},
        "seeds": SEEDS,
        "n_bootstrap": N_BOOT,
        "vlm_reference": {
            "source": "outputs/vlm_concept_boost/results.json",
            "dino_plus_vlm_dWF1_seedmean_ci95": [-0.006708777616806011, -0.00841776632038311, -0.004999788913228911],
            "note": "UNFAITHFUL VLM concepts HURT the backbone (dWF1 = -0.0067).",
        },
        "cond1_morpho_alone": {
            "wf1": agg("wf1"), "macrof1": agg("macrof1"), "acc": agg("acc"),
            "per_class_f1": agg_per_class(),
        },
        "cond2_faithful_boost": {
            "wf1_dino": agg("wf1")["baseline_dino"],
            "wf1_dino_plus_morpho": agg("wf1")["dino_plus_morpho"],
            "delta_morpho_vs_baseline_pooled": pooled_delta,
        },
        "cond3_faithful_cbm": cbm_agg,
        "cond4_concept_intervention": interv_agg,
        "per_seed": per_seed,
        "cbm_per_seed": cbm_per_seed,
        "interv_per_seed": interv_per_seed,
        "note": "dino_plus_morpho = [DinoBloom-B 768d (+) 11 measured concepts]. "
                "morpho_alone = 11 measured concepts only (fully transparent). "
                "CBM = x->concept-head(Ridge)->predicted concepts->class-head(LogReg); "
                "decisions trace to concepts. Intervention sets a concept on the "
                "bottleneck to true measured / forced value and reads the prob shift. "
                "Cells restricted to those with BOTH a DinoBloom feature and a morphometry "
                "measurement for a fair paired comparison. LogReg, class_weight=balanced.",
    }
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[done] wrote {OUT_DIR/'results.json'}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
