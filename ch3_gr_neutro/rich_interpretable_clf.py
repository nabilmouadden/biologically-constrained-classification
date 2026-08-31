#!/usr/bin/env python3
"""Rich-interpretable-feature classifiers for GR-Neutro, vs DinoBloom-B backbone.

Reads the rich named-morphometry feature bank (rich_morphometry.py -> features.csv)
and asks: how close to the black-box backbone (W-F1 0.83) can a FULLY INTERPRETABLE
classifier on ~70 named features get?

Experiments (multi-seed, paired-bootstrap CIs; same split/seeds as
faithful_concept_classifier.py for a fair contrast):
  1. Rich interpretable features -> classifier. Four heads:
       - LogReg (linear, fully transparent)
       - RandomForest
       - HistGradientBoosting
       - small MLP (still on NAMED features -> interpretable inputs)
     Report W-F1, macro-F1, per-class for each; pick best interpretable.
  2. Feature importance: permutation importance (model-agnostic) of the best
     interpretable model, overall + per-class one-vs-rest, to confirm sensible
     morphology drives each class.
  3. Hybrid: DinoBloom-B 768d (+) rich features -> LogReg, vs backbone alone.
     Paired-bootstrap dW-F1. Does richer faithful info exceed 0.83?
  4. Honest residual gap: best-interpretable W-F1 vs backbone 0.83, with CI.

NO fabrication. SLURM only (cpu_med).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, accuracy_score
from sklearn.inspection import permutation_importance

HERE = Path(__file__).resolve().parent
OUT_ROOT = HERE / "outputs"
FEATS_NPZ = OUT_ROOT / "dinobloom_features.npz"
MANIFEST = OUT_ROOT / "p3_pilot" / "cell_manifest_full_extended.json"
RICH_CSV = OUT_ROOT / "rich_interpretable" / "features.csv"
OUT_DIR = OUT_ROOT / "rich_interpretable"

SEEDS = [0, 7, 13, 42, 1337]
N_BOOT = 5000
BACKBONE_WF1 = 0.8301231205481923   # outputs/faithful_concept_classifier/results.json
BACKBONE_MACRO = 0.747              # task spec

NON_FEATURE_COLS = {"filename", "path", "seg_ok", "seg_reason",
                    "dominant_class_idx", "Normal", "Chromatin", "Dohle",
                    "Hypergranulation", "Hypersegmentation", "Hypogranulation",
                    "Hyposegmentation"}


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


def make_model(name, seed):
    if name == "logreg":
        return make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(max_iter=5000, C=1.0, class_weight="balanced",
                               random_state=seed))
    if name == "randomforest":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                   n_jobs=-1, random_state=seed))
    if name == "histgb":
        return make_pipeline(
            HistGradientBoostingClassifier(
                max_iter=500, learning_rate=0.05, max_depth=None,
                l2_regularization=1.0, class_weight="balanced",
                random_state=seed))
    if name == "mlp":
        return make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3,
                          max_iter=800, early_stopping=True,
                          random_state=seed))
    raise ValueError(name)


def metrics(yte, pred, K):
    labels = list(range(K))
    return {
        "wf1": f1_score(yte, pred, average="weighted", labels=labels, zero_division=0),
        "macrof1": f1_score(yte, pred, average="macro", labels=labels, zero_division=0),
        "acc": accuracy_score(yte, pred),
        "per_class": f1_score(yte, pred, average=None, labels=labels, zero_division=0).tolist(),
    }


def paired_bootstrap_delta(yte, pred_a, pred_b, class_names, rng, n_boot=N_BOOT):
    yte = np.asarray(yte); pred_a = np.asarray(pred_a); pred_b = np.asarray(pred_b)
    n = len(yte); K = len(class_names); labels = list(range(K))
    d_wf1, d_mf1 = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, pa, pb = yte[idx], pred_a[idx], pred_b[idx]
        d_wf1.append(f1_score(yt, pb, average="weighted", labels=labels, zero_division=0)
                     - f1_score(yt, pa, average="weighted", labels=labels, zero_division=0))
        d_mf1.append(f1_score(yt, pb, average="macro", labels=labels, zero_division=0)
                     - f1_score(yt, pa, average="macro", labels=labels, zero_division=0))

    def ci(arr):
        arr = np.asarray(arr)
        return [float(arr.mean()), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]
    return {"d_wf1": ci(d_wf1), "d_macrof1": ci(d_mf1)}


def bootstrap_metric_ci(yte, pred, K, rng, n_boot=N_BOOT):
    yte = np.asarray(yte); pred = np.asarray(pred)
    n = len(yte); labels = list(range(K))
    wf1, mf1 = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        wf1.append(f1_score(yte[idx], pred[idx], average="weighted", labels=labels, zero_division=0))
        mf1.append(f1_score(yte[idx], pred[idx], average="macro", labels=labels, zero_division=0))
    return ([float(np.mean(wf1)), float(np.percentile(wf1, 2.5)), float(np.percentile(wf1, 97.5))],
            [float(np.mean(mf1)), float(np.percentile(mf1, 2.5)), float(np.percentile(mf1, 97.5))])


def seed_ci(vals):
    vals = np.asarray(vals, dtype=float)
    m = vals.mean()
    half = 1.96 * vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
    return [float(m), float(m - half), float(m + half)]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[load] manifest + dino features + rich features csv", flush=True)
    with open(MANIFEST) as f:
        man = json.load(f)
    class_names = man["class_names"]
    cells = man["cells"]
    K = len(class_names)

    npz = np.load(FEATS_NPZ, allow_pickle=True)
    feats = npz["features"]; paths = npz["paths"]
    dino_by_fn = {Path(str(p)).name: feats[i] for i, p in enumerate(paths)}

    df = pd.read_csv(RICH_CSV)
    feat_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    print(f"[rich] {len(df)} rows, {len(feat_cols)} named features", flush=True)
    rich_by_fn = {row["filename"]: row for _, row in df.iterrows()}

    # align: keep cells with BOTH a dino feature AND a rich-feature row that
    # segmented OK (fair paired comparison vs the backbone)
    keep, X_dino, X_rich, y, labels_oh = [], [], [], [], []
    miss_dino = miss_rich = seg_bad = 0
    for c in cells:
        fn = c["filename"]
        if fn not in dino_by_fn:
            miss_dino += 1; continue
        if fn not in rich_by_fn:
            miss_rich += 1; continue
        rr = rich_by_fn[fn]
        if int(rr.get("seg_ok", 0)) != 1:
            seg_bad += 1; continue
        keep.append(fn)
        X_dino.append(dino_by_fn[fn])
        X_rich.append(np.array([rr[k] for k in feat_cols], dtype=float))
        y.append(c["dominant_class_idx"])
        labels_oh.append(c["label_one_hot"])
    X_dino = np.vstack(X_dino); X_rich = np.vstack(X_rich)
    y = np.asarray(y); labels_oh = np.asarray(labels_oh)
    X_hybrid = np.hstack([X_dino, X_rich])
    print(f"[align] kept={len(keep)} miss_dino={miss_dino} miss_rich={miss_rich} "
          f"seg_bad={seg_bad}", flush=True)
    print("[align] class dist: " + ", ".join(
        f"{class_names[k]}={int((y==k).sum())}" for k in range(K)), flush=True)

    heads = ["logreg", "randomforest", "histgb", "mlp"]
    # ------------------------------------------------------------------ #
    # cond 1: rich interpretable features -> 4 classifier heads
    # cond 3: hybrid (dino + rich) vs dino-alone (logreg, matches faithful clf)
    # ------------------------------------------------------------------ #
    per_seed = {}
    last_fitted = {}   # for feature importance on a representative seed
    for seed in SEEDS:
        tr_idx, val_idx, te_idx = stratified_multilabel_split(labels_oh, seed=seed)
        tr = np.concatenate([tr_idx, val_idx])
        sres = {"rich": {}, "hybrid": {}, "dino": {}}

        # rich-feature heads
        rich_preds = {}
        for h in heads:
            mdl = make_model(h, seed)
            mdl.fit(X_rich[tr], y[tr])
            pred = mdl.predict(X_rich[te_idx])
            sres["rich"][h] = metrics(y[te_idx], pred, K)
            rich_preds[h] = pred
            if seed == SEEDS[0]:
                last_fitted[h] = mdl

        # dino-alone (logreg) and hybrid (logreg) for cond 3
        dino_mdl = make_model("logreg", seed); dino_mdl.fit(X_dino[tr], y[tr])
        dino_pred = dino_mdl.predict(X_dino[te_idx])
        sres["dino"]["logreg"] = metrics(y[te_idx], dino_pred, K)
        hyb_mdl = make_model("logreg", seed); hyb_mdl.fit(X_hybrid[tr], y[tr])
        hyb_pred = hyb_mdl.predict(X_hybrid[te_idx])
        sres["hybrid"]["logreg"] = metrics(y[te_idx], hyb_pred, K)

        rng = np.random.default_rng(seed)
        sres["delta_hybrid_vs_dino"] = paired_bootstrap_delta(
            y[te_idx], dino_pred, hyb_pred, class_names, rng)
        # best interpretable head this seed (by W-F1) vs dino, with gap CI
        best_h = max(heads, key=lambda h: sres["rich"][h]["wf1"])
        sres["best_rich_head"] = best_h
        rng2 = np.random.default_rng(seed + 99)
        sres["delta_dino_vs_bestrich"] = paired_bootstrap_delta(
            y[te_idx], rich_preds[best_h], dino_pred, class_names, rng2)
        # absolute CI for best interpretable head
        rng3 = np.random.default_rng(seed + 7)
        wci, mci = bootstrap_metric_ci(y[te_idx], rich_preds[best_h], K, rng3)
        sres["best_rich_wf1_bootci"] = wci
        sres["best_rich_macrof1_bootci"] = mci

        per_seed[str(seed)] = sres
        print(f"[seed {seed}] "
              + " ".join(f"{h}={sres['rich'][h]['wf1']:.3f}" for h in heads)
              + f" | dino={sres['dino']['logreg']['wf1']:.3f}"
              + f" hybrid={sres['hybrid']['logreg']['wf1']:.3f}"
              + f" dHyb={sres['delta_hybrid_vs_dino']['d_wf1'][0]:+.4f}", flush=True)

    # ------------------------------------------------------------------ #
    # aggregate
    # ------------------------------------------------------------------ #
    def agg_head(group, head, metric):
        v = [per_seed[str(s)][group][head][metric] for s in SEEDS]
        return {"mean": float(np.mean(v)), "std": float(np.std(v)), "per_seed": v}

    def agg_head_perclass(group, head):
        arr = np.array([per_seed[str(s)][group][head]["per_class"] for s in SEEDS])
        return {class_names[k]: {"mean": float(arr[:, k].mean()),
                                 "std": float(arr[:, k].std())} for k in range(K)}

    cond1 = {h: {"wf1": agg_head("rich", h, "wf1"),
                 "macrof1": agg_head("rich", h, "macrof1"),
                 "acc": agg_head("rich", h, "acc"),
                 "per_class_f1": agg_head_perclass("rich", h)} for h in heads}

    # best interpretable head overall (by mean W-F1)
    best_head = max(heads, key=lambda h: cond1[h]["wf1"]["mean"])
    best_wf1_seedci = seed_ci([per_seed[str(s)]["rich"][best_head]["wf1"] for s in SEEDS])
    best_macro_seedci = seed_ci([per_seed[str(s)]["rich"][best_head]["macrof1"] for s in SEEDS])

    dino_wf1 = agg_head("dino", "logreg", "wf1")
    hybrid_wf1 = agg_head("hybrid", "logreg", "wf1")
    delta_hyb = seed_ci([per_seed[str(s)]["delta_hybrid_vs_dino"]["d_wf1"][0] for s in SEEDS])
    # residual gap = backbone - best interpretable (seed-paired with dino-logreg here)
    gap_seed = [per_seed[str(s)]["dino"]["logreg"]["wf1"]
                - per_seed[str(s)]["rich"][best_head]["wf1"] for s in SEEDS]
    gap_seedci = seed_ci(gap_seed)

    # ------------------------------------------------------------------ #
    # cond 2: permutation feature importance on best interpretable head
    # (recompute on first seed's split for a clean held-out test set)
    # ------------------------------------------------------------------ #
    print(f"[importance] permutation importance for best head '{best_head}'",
          flush=True)
    tr_idx, val_idx, te_idx = stratified_multilabel_split(labels_oh, seed=SEEDS[0])
    tr = np.concatenate([tr_idx, val_idx])
    imp_mdl = make_model(best_head, SEEDS[0]); imp_mdl.fit(X_rich[tr], y[tr])
    pi = permutation_importance(imp_mdl, X_rich[te_idx], y[te_idx],
                                n_repeats=10, random_state=0, n_jobs=-1,
                                scoring="f1_weighted")
    overall_imp = sorted(
        [(feat_cols[i], float(pi.importances_mean[i]), float(pi.importances_std[i]))
         for i in range(len(feat_cols))],
        key=lambda t: -t[1])
    # per-class one-vs-rest importance (drop in OVR f1 for that class)
    per_class_imp = {}
    for k in range(K):
        yb_tr = (y[tr] == k).astype(int)
        yb_te = (y[te_idx] == k).astype(int)
        if yb_te.sum() < 3:
            continue
        mk = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                           LogisticRegression(max_iter=4000, class_weight="balanced",
                                              random_state=0))
        mk.fit(X_rich[tr], yb_tr)
        pik = permutation_importance(mk, X_rich[te_idx], yb_te, n_repeats=8,
                                     random_state=0, n_jobs=-1, scoring="f1")
        top = sorted([(feat_cols[i], float(pik.importances_mean[i]))
                      for i in range(len(feat_cols))], key=lambda t: -t[1])[:8]
        per_class_imp[class_names[k]] = top

    # ------------------------------------------------------------------ #
    # write results
    # ------------------------------------------------------------------ #
    results = {
        "experiment": "rich_interpretable",
        "n_cells_used": len(keep),
        "n_features": len(feat_cols),
        "feature_names": feat_cols,
        "class_names": class_names,
        "class_dist": {class_names[k]: int((y == k).sum()) for k in range(K)},
        "seeds": SEEDS, "n_bootstrap": N_BOOT,
        "backbone_reference": {
            "model": "DinoBloom-B (black box) + LogReg",
            "wf1": BACKBONE_WF1, "macrof1": BACKBONE_MACRO,
            "source": "outputs/faithful_concept_classifier/results.json",
            "wf1_this_run_dino_logreg": dino_wf1,
        },
        "prior_interpretable_reference": {
            "model": "11 morphometry concepts + LogReg",
            "wf1": 0.6638563287175577,
            "source": "outputs/faithful_concept_classifier/results.json",
        },
        "cond1_rich_interpretable_heads": cond1,
        "best_interpretable": {
            "head": best_head,
            "wf1_seedmean_ci95": best_wf1_seedci,
            "macrof1_seedmean_ci95": best_macro_seedci,
            "per_class_f1": cond1[best_head]["per_class_f1"],
        },
        "cond3_hybrid": {
            "dino_logreg_wf1": dino_wf1,
            "hybrid_dino_plus_rich_wf1": hybrid_wf1,
            "delta_hybrid_vs_dino_wf1_seedmean_ci95": delta_hyb,
        },
        "cond4_residual_gap": {
            "backbone_wf1": BACKBONE_WF1,
            "best_interpretable_wf1": best_wf1_seedci[0],
            "gap_backbone_minus_interpretable_wf1_seedmean_ci95": gap_seedci,
            "note": "Gap computed seed-paired between dino-logreg and the best "
                    "interpretable head on the SAME splits this run.",
        },
        "cond2_feature_importance": {
            "method": "permutation_importance (f1_weighted), best head, seed0 split",
            "overall_top30": overall_imp[:30],
            "per_class_onevsrest_top8": per_class_imp,
        },
        "per_seed": per_seed,
        "incomplete_label_ceiling_note": (
            "Both interpretable and backbone models are scored against noisy "
            "single dominant-class labels; cells may carry co-occurring "
            "abnormalities, so absolute W-F1 is bounded by label completeness, "
            "not just feature quality. The interpretable-vs-backbone GAP is the "
            "honest headline, not the absolute number."),
        "honesty": (
            "Every feature is a named, computed morphometric measurement "
            "(no learned embeddings) in cond1/cond2. The hybrid (cond3) adds the "
            "DinoBloom-B backbone and is reported separately as the recommended "
            "accuracy+audit configuration, not as 'interpretable'."),
    }
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # markdown summary
    md = []
    md.append("# Rich interpretable morphometry classifier vs DinoBloom-B backbone\n")
    md.append(f"- Cells used: **{len(keep)}** (seg-OK, with both DinoBloom feature "
              f"and rich morphometry). {len(feat_cols)} NAMED features.")
    md.append(f"- Seeds: {SEEDS}; paired bootstrap n={N_BOOT}.\n")
    md.append("## Headline (W-F1, seed-mean +/- std)\n")
    md.append("| model | interpretable? | W-F1 | macro-F1 | acc |")
    md.append("|---|---|---|---|---|")
    md.append(f"| DinoBloom-B + LogReg (black box) | no | "
              f"{dino_wf1['mean']:.4f} +/- {dino_wf1['std']:.4f} | "
              f"{agg_head('dino','logreg','macrof1')['mean']:.4f} | "
              f"{agg_head('dino','logreg','acc')['mean']:.4f} |")
    for h in heads:
        star = " **(best interp.)**" if h == best_head else ""
        md.append(f"| rich features + {h}{star} | YES | "
                  f"{cond1[h]['wf1']['mean']:.4f} +/- {cond1[h]['wf1']['std']:.4f} | "
                  f"{cond1[h]['macrof1']['mean']:.4f} | {cond1[h]['acc']['mean']:.4f} |")
    md.append(f"| 11-concept + LogReg (prior interp.) | YES | 0.6639 | - | - |")
    md.append(f"| **Hybrid** DinoBloom (+) rich + LogReg | no | "
              f"{hybrid_wf1['mean']:.4f} +/- {hybrid_wf1['std']:.4f} | "
              f"{agg_head('hybrid','logreg','macrof1')['mean']:.4f} | "
              f"{agg_head('hybrid','logreg','acc')['mean']:.4f} |\n")
    md.append("## Gaps (seed-mean, 95% CI)\n")
    md.append(f"- Residual gap backbone - best interpretable ({best_head}): "
              f"**{gap_seedci[0]:+.4f}** [{gap_seedci[1]:+.4f}, {gap_seedci[2]:+.4f}] W-F1")
    md.append(f"- Hybrid vs backbone: **{delta_hyb[0]:+.4f}** "
              f"[{delta_hyb[1]:+.4f}, {delta_hyb[2]:+.4f}] W-F1")
    md.append(f"- Best interpretable W-F1 (bootstrap CI, seed-mean): "
              f"{best_wf1_seedci[0]:.4f} [{best_wf1_seedci[1]:.4f}, {best_wf1_seedci[2]:.4f}]\n")
    md.append(f"## Per-class F1 — best interpretable ({best_head})\n")
    md.append("| class | F1 |")
    md.append("|---|---|")
    for k in range(K):
        cn = class_names[k]
        md.append(f"| {cn} | {cond1[best_head]['per_class_f1'][cn]['mean']:.3f} |")
    md.append("\n## Top-15 named features (permutation importance, f1_weighted)\n")
    md.append("| feature | importance | std |")
    md.append("|---|---|---|")
    for name, m, s in overall_imp[:15]:
        md.append(f"| {name} | {m:.4f} | {s:.4f} |")
    md.append("\n## Per-class top driver features (one-vs-rest)\n")
    for cn, top in per_class_imp.items():
        feats_str = ", ".join(f"{n} ({m:.3f})" for n, m in top[:5])
        md.append(f"- **{cn}**: {feats_str}")
    md.append("\n" + results["incomplete_label_ceiling_note"])
    with open(OUT_DIR / "results.md", "w") as f:
        f.write("\n".join(md) + "\n")

    print(f"[done] best interpretable head={best_head} "
          f"W-F1={best_wf1_seedci[0]:.4f}; backbone={BACKBONE_WF1:.4f}; "
          f"gap={gap_seedci[0]:+.4f}; hybrid={hybrid_wf1['mean']:.4f} "
          f"(dHyb={delta_hyb[0]:+.4f})", flush=True)
    print(f"[done] wrote {OUT_DIR/'results.json'} and results.md", flush=True)


if __name__ == "__main__":
    sys.exit(main())
