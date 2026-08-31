"""Shared frozen-feature multi-label probe for the CNN-vs-DinoBloom comparison.

ONE code path, run once per encoder (DinoBloom-B CLS 768-d, ResNet50 pool 2048-d).
The encoder is the ONLY variable. Everything else — split, head, threshold
procedure, seeds, metrics — is byte-identical between arms.

Pipeline (per seed in {0,7,13,42,1337}):
  1. Load the encoder's GR-Neutro feature bank; join labels by basename from
     annotations.csv (same join as train_loco.load_gr_neutro).
  2. stratified_multilabel_split(labels, test_size=0.10, val_size=0.10, seed) —
     the paper's split. (data.py, identical logic.)
  3. StandardScaler fit on TRAIN features only; applied to val/test/external.
  4. One-vs-rest multi-label LogisticRegression (7 classes), class_weight balanced,
     max_iter 2000. Trained on TRAIN.
  5. Per-class threshold swept on VAL to maximise macro-F1; frozen.
  6. Internal metrics on TEST: macro-F1, weighted-F1 (at tuned thresholds),
     abnormal-vs-normal AUROC (score = 1 - P(Normal); label = not-Normal).
  7. External metrics on Barrera evaluable subset (NEU normal vs HYP+DB abnormal,
     OOV excluded): abn-vs-normal AUROC, abnormal recall (fraction of abnormal
     scored not-Normal above the fixed Normal threshold), specificity (Normal
     recall). Thresholds are the FIXED GR-val ones — NO external tuning.

Aggregate: mean +- sd across seeds and a paired-bootstrap 95% CI (one cell
resample applied to all seeds, metric averaged across seeds per replicate),
computed on the external evaluable subset (matches M1's paired-bootstrap rule).

Usage:
  python probe_cnn_vs_dinobloom.py --encoder dinobloom_b
  python probe_cnn_vs_dinobloom.py --encoder resnet50
Both write into outputs/cnn_vs_dinobloom/<encoder>.json; a final --aggregate call
merges them into results.json + the comparison table rows.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import numpy as np

WORKDIR = Path("/gpfs/workdir/mouaddenn")
HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs" / "cnn_vs_dinobloom"

GR_ANN = WORKDIR / "data" / "gr_neutro_extended" / "annotations.csv"

GR_CLASSES = ["Normal", "Chromatin", "Dohle", "Hypergranulation",
              "Hypersegmentation", "Hypogranulation", "Hyposegmentation"]
NORMAL_IDX = 0
SEEDS = [0, 7, 13, 42, 1337]

# Encoder -> (GR feature npz, Barrera feature npz)
BANKS = {
    "dinobloom_b": (
        WORKDIR / "thesis/ch3_gr_neutro/outputs/dinobloom_features.npz",
        WORKDIR / "data/barrera_neunn/features/dinobloom_b_cls.npz",
    ),
    "resnet50": (
        WORKDIR / "thesis/ch3_gr_neutro/outputs/resnet50_features.npz",
        WORKDIR / "data/barrera_neunn/features/resnet50_pool.npz",
    ),
}

# Barrera source-class -> GR-Neutro mapping for the evaluable abnormal-vs-normal
# subset (identical to M1). Case-robust match on the zip folder names.
BARRERA_NORMAL = {"NEU"}
BARRERA_ABNORMAL = {"HYP", "DB"}   # Hypogranulation, Dohle
BARRERA_OOV = {"CRY", "HJBLI", "GBI", "BAC"}


# ----------------------------------------------------------------------------
def stratified_multilabel_split(labels_np, test_size=0.10, val_size=0.10, seed=42):
    """Identical to data.py::stratified_multilabel_split."""
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


def load_gr(gr_npz):
    d = np.load(gr_npz, allow_pickle=True)
    feats = d["features"].astype(np.float32)
    paths = d["paths"]
    ann = {}
    with open(GR_ANN) as f:
        r = csv.reader(f); hdr = next(r)
        csv_classes = hdr[2:]
        for row in r:
            ann[row[0]] = [int(x) for x in row[2:]]
    assert csv_classes == GR_CLASSES, f"class mismatch: {csv_classes}"
    labels = np.zeros((len(paths), len(GR_CLASSES)), dtype=np.int64)
    for i, p in enumerate(paths):
        labels[i] = ann[Path(str(p)).name]
    return feats, labels


def load_barrera(bar_npz):
    d = np.load(bar_npz, allow_pickle=True)
    feats = d["features"].astype(np.float32)
    labels = d["labels"].astype(np.int64)
    class_names = [str(c) for c in d["class_names"]]
    src = np.asarray([class_names[i] for i in labels])
    # evaluable subset: NEU (normal) vs HYP/DB (abnormal); OOV excluded.
    def bucket(s):
        su = s.upper()
        if su in BARRERA_NORMAL: return "normal"
        if su in BARRERA_ABNORMAL: return "abnormal"
        return "oov"
    bkt = np.asarray([bucket(s) for s in src])
    keep = bkt != "oov"
    return feats[keep], (bkt[keep] == "abnormal").astype(np.int64)


def sweep_thresholds(val_probs, val_labels):
    """Per-class threshold maximising macro-F1 contribution (one-vs-rest)."""
    from sklearn.metrics import f1_score
    K = val_labels.shape[1]
    grid = np.arange(0.05, 0.96, 0.05)
    thr = np.full(K, 0.5)
    for k in range(K):
        best_t, best_f = 0.5, -1.0
        for t in grid:
            f = f1_score(val_labels[:, k], (val_probs[:, k] >= t).astype(int),
                         zero_division=0)
            if f > best_f:
                best_f, best_t = f, t
        thr[k] = best_t
    return thr


def run_seed(encoder, seed):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, roc_auc_score

    gr_npz, bar_npz = BANKS[encoder]
    Xgr, Ygr = load_gr(gr_npz)
    tr, va, te = stratified_multilabel_split(Ygr, 0.10, 0.10, seed)

    scaler = StandardScaler().fit(Xgr[tr])
    Xtr, Xva, Xte = scaler.transform(Xgr[tr]), scaler.transform(Xgr[va]), scaler.transform(Xgr[te])

    K = Ygr.shape[1]
    # per-class logistic regression (one-vs-rest multi-label). Keep the fitted
    # classifiers so the SAME Normal classifier drives internal and external
    # abnormal-vs-normal decisions (no separate refit -> identical surface).
    val_probs = np.zeros((len(va), K))
    test_probs = np.zeros((len(te), K))
    clfs = [None] * K
    for k in range(K):
        ytr = Ygr[tr, k]
        if ytr.sum() == 0 or ytr.sum() == len(ytr):
            # degenerate class in this split: constant prob
            p = float(ytr.mean())
            val_probs[:, k] = p; test_probs[:, k] = p
            continue
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
        clf.fit(Xtr, ytr)
        clfs[k] = clf
        val_probs[:, k] = clf.predict_proba(Xva)[:, 1]
        test_probs[:, k] = clf.predict_proba(Xte)[:, 1]

    thr = sweep_thresholds(val_probs, Ygr[va])

    # internal metrics at tuned thresholds
    test_pred = (test_probs >= thr[None, :]).astype(int)
    macro_f1 = float(f1_score(Ygr[te], test_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(Ygr[te], test_pred, average="weighted", zero_division=0))
    # internal abnormal-vs-normal AUROC: score = 1 - P(Normal), label = not-Normal
    int_abn_label = (Ygr[te, NORMAL_IDX] == 0).astype(int)
    int_abn_score = 1.0 - test_probs[:, NORMAL_IDX]
    int_auroc = (float(roc_auc_score(int_abn_label, int_abn_score))
                 if 0 < int_abn_label.sum() < len(int_abn_label) else float("nan"))

    # ---- external (Barrera evaluable subset) ----
    Xbar, ybar_abn = load_barrera(bar_npz)  # ybar_abn: 1=abnormal, 0=normal
    Xbar = scaler.transform(Xbar)
    # Reuse the SAME Normal classifier trained above (no refit) so the external
    # decision surface is byte-identical to the internal one.
    clfN = clfs[NORMAL_IDX]
    assert clfN is not None, "Normal class degenerate in this split"
    p_normal_bar = clfN.predict_proba(Xbar)[:, 1]
    ext_score = 1.0 - p_normal_bar             # not-Normal score
    ext_auroc = float(roc_auc_score(ybar_abn, ext_score))
    # decision at FIXED GR-val Normal threshold: fires abnormal if p_normal < thr_Normal
    fired_abnormal = (p_normal_bar < thr[NORMAL_IDX]).astype(int)
    abn_recall = float(fired_abnormal[ybar_abn == 1].mean())          # sensitivity
    specificity = float((1 - fired_abnormal)[ybar_abn == 0].mean())   # normal recall

    return dict(
        seed=seed,
        internal_macro_f1=macro_f1,
        internal_weighted_f1=weighted_f1,
        internal_abn_auroc=int_auroc,
        external_abn_auroc=ext_auroc,
        external_abn_recall=abn_recall,
        external_specificity=specificity,
        thr_normal=float(thr[NORMAL_IDX]),
        # per-cell external arrays for paired bootstrap
        _ext_label=ybar_abn.tolist(),
        _ext_score=ext_score.tolist(),
        _ext_fired=fired_abnormal.tolist(),
    )


def run_encoder(encoder):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_seed = [run_seed(encoder, s) for s in SEEDS]
    # scalar aggregate
    keys = ["internal_macro_f1", "internal_weighted_f1", "internal_abn_auroc",
            "external_abn_auroc", "external_abn_recall", "external_specificity"]
    agg = {}
    for k in keys:
        vals = np.array([s[k] for s in per_seed], dtype=float)
        agg[k] = dict(mean=float(np.nanmean(vals)), sd=float(np.nanstd(vals)))
    # paired bootstrap on external evaluable subset (2000 replicates)
    labels = np.array(per_seed[0]["_ext_label"])
    n = len(labels)
    assert all(np.array_equal(np.array(s["_ext_label"]), labels) for s in per_seed), \
        "external cell order must be identical across seeds"
    scores = [np.array(s["_ext_score"]) for s in per_seed]
    fired = [np.array(s["_ext_fired"]) for s in per_seed]
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    B = 2000
    boot = {"external_abn_auroc": [], "external_abn_recall": [], "external_specificity": []}
    pos = np.where(labels == 1)[0]; neg = np.where(labels == 0)[0]
    for _ in range(B):
        idx = rng.integers(0, n, n)
        li = labels[idx]
        if li.sum() == 0 or li.sum() == n:
            continue
        auroc_s, rec_s, spec_s = [], [], []
        for si in range(len(SEEDS)):
            auroc_s.append(roc_auc_score(li, scores[si][idx]))
            fi = fired[si][idx]
            rec_s.append(fi[li == 1].mean())
            spec_s.append((1 - fi)[li == 0].mean())
        boot["external_abn_auroc"].append(np.mean(auroc_s))
        boot["external_abn_recall"].append(np.mean(rec_s))
        boot["external_specificity"].append(np.mean(spec_s))
    ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
          for k, v in boot.items()}

    # strip heavy per-cell arrays from saved per-seed
    clean = [{k: v for k, v in s.items() if not k.startswith("_")} for s in per_seed]
    out = dict(encoder=encoder, seeds=SEEDS, per_seed=clean, aggregate=agg,
               external_ci=ci, n_external=int(n),
               n_external_abnormal=int(labels.sum()),
               n_external_normal=int((labels == 0).sum()))
    p = OUT_DIR / f"{encoder}.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"[save] {p}", flush=True)
    print(json.dumps(agg, indent=2), flush=True)
    return out


def aggregate():
    rows = {}
    for enc in ["dinobloom_b", "resnet50"]:
        p = OUT_DIR / f"{enc}.json"
        if not p.exists():
            print(f"[skip] {p} missing"); continue
        rows[enc] = json.loads(p.read_text())
    merged = OUT_DIR / "results.json"
    # compute the external-drop story: internal vs external abn AUROC
    summary = {}
    for enc, r in rows.items():
        a = r["aggregate"]
        summary[enc] = dict(
            internal_abn_auroc=a["internal_abn_auroc"]["mean"],
            internal_macro_f1=a["internal_macro_f1"]["mean"],
            external_abn_auroc=a["external_abn_auroc"]["mean"],
            external_abn_recall=a["external_abn_recall"]["mean"],
            external_specificity=a["external_specificity"]["mean"],
            external_drop=(a["internal_abn_auroc"]["mean"]
                          - a["external_abn_auroc"]["mean"]),
        )
    (OUT_DIR / "results.json").write_text(
        json.dumps(dict(per_encoder=rows, summary=summary), indent=2))
    print(f"[save] {merged}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", choices=list(BANKS.keys()))
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()
    if args.aggregate:
        aggregate()
    else:
        run_encoder(args.encoder)


if __name__ == "__main__":
    main()
