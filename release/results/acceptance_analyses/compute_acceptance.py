#!/usr/bin/env python
"""Reproducer for the BJH paper acceptance-plan analyses (CPU only).

Recomputes, from the frozen test-set predictions of the representative seed
(cbm_joint s42), every acceptance-plan number cited in the paper:
  - per_class_cis.json       : per-class precision/recall/F1 + 1000x bootstrap 95% CIs
  - per_class_calibration.json: per-class ECE / Brier + pooled ECE 0.054 / Brier 0.029
  - clinical_utility.json    : abnormal-vs-normal AUROC 0.997, decision curve, triage yield

The shipped JSONs in this directory are the outputs of exactly this script; they
are what the paper's tables/figures are drawn from. This script is included so
the pipeline is inspectable and rerunnable given the three input artifacts.

INPUTS (not shipped; regenerable from the released model + GR-Neutro test split):
  $ACC_BASE/predictions.pt       test dict {class_logits[N,7], labels[N,7]}, class_names
  $ACC_BASE/eval.json            classification_tuned.thresholds[7]
  $ACC_BASE/split_filelist.json  train/val/test bare filenames (leakage audit)
Set ACC_BASE to the directory holding those three files, then run:
  python compute_acceptance.py     # writes the JSONs next to this script
"""
import os, json
import numpy as np
import torch

np.random.seed(42)

BASE = os.environ.get("ACC_BASE", os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)

# This reproducer additionally needs `matplotlib` (not in the core CPU
# requirements.txt) and three unshipped inputs (predictions.pt, eval.json,
# split_filelist.json) that carry per-cell test predictions. The three JSONs it
# emits are already shipped; the paper numbers reproduce via
# `results/reproduce_paper.py` without running this script.
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as e:
    raise SystemExit(
        "compute_acceptance.py needs matplotlib (pip install matplotlib) plus the "
        "unshipped predictions.pt/eval.json/split_filelist.json inputs. The shipped "
        "JSONs it produces are reproduced by results/reproduce_paper.py without it."
    ) from e

# ---------------------------------------------------------------- load
d = torch.load(os.path.join(BASE, "predictions.pt"), map_location="cpu", weights_only=False)
class_names = d["class_names"]
t = d["test"]
logits = t["class_logits"].numpy().astype(np.float64)          # [438,7]
labels = t["labels"].numpy().astype(np.int64)                  # [438,7]
probs = 1.0 / (1.0 + np.exp(-logits))                          # sigmoid, multi-label
N, C = labels.shape

eval_json = json.load(open(os.path.join(BASE, "eval.json")))
thresholds = np.array(eval_json["classification_tuned"]["thresholds"], dtype=np.float64)  # [7]
assert thresholds.shape[0] == C

filelist = json.load(open(os.path.join(BASE, "split_filelist.json")))
test_files = filelist["test_files"]
train_files = filelist["train_files"]
val_files = filelist["val_files"]
assert len(test_files) == N

preds = (probs >= thresholds[None, :]).astype(np.int64)         # [438,7]


# ================================================================ M4a
# Per-class precision/recall/F1 with 1000x stratified bootstrap 95% CIs.
def prf(y, p):
    tp = int(((y == 1) & (p == 1)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    rec = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    if np.isnan(prec) or np.isnan(rec) or (prec + rec) == 0:
        f1 = np.nan if (np.isnan(prec) or np.isnan(rec)) else 0.0
    else:
        f1 = 2 * prec * rec / (prec + rec)
    return prec, rec, f1, tp, fp, fn


B = 1000
idx_all = np.arange(N)
m4a = {"n_test": N, "n_bootstrap": B, "classes": {}}

for c in range(C):
    y = labels[:, c]
    p = preds[:, c]
    pos_idx = idx_all[y == 1]
    neg_idx = idx_all[y == 0]
    n_pos, n_neg = len(pos_idx), len(neg_idx)
    prec, rec, f1, tp, fp, fn = prf(y, p)

    bp, br, bf = [], [], []
    for _ in range(B):
        # stratified resample: keep positive/negative counts fixed
        rs = np.concatenate([
            np.random.choice(pos_idx, n_pos, replace=True) if n_pos > 0 else np.array([], int),
            np.random.choice(neg_idx, n_neg, replace=True) if n_neg > 0 else np.array([], int),
        ])
        yy, pp = y[rs], p[rs]
        pr, re, ff, *_ = prf(yy, pp)
        bp.append(pr); br.append(re); bf.append(ff)

    def ci(arr):
        a = np.array(arr, dtype=np.float64)
        a = a[~np.isnan(a)]
        if a.size == 0:
            return [None, None]
        return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]

    m4a["classes"][class_names[c]] = {
        "n_pos": int(n_pos), "n_neg": int(n_neg),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": None if np.isnan(prec) else float(prec),
        "recall": None if np.isnan(rec) else float(rec),
        "f1": None if np.isnan(f1) else float(f1),
        "recall_fraction": f"{tp}/{tp+fn}",
        "precision_ci95": ci(bp),
        "recall_ci95": ci(br),
        "f1_ci95": ci(bf),
    }

json.dump(m4a, open(os.path.join(OUT, "per_class_cis.json"), "w"), indent=2)
print("[M4a] written per_class_cis.json")
for cn in class_names:
    r = m4a["classes"][cn]
    print(f"  {cn:16s} P={r['precision']:.3f}{r['precision_ci95']} R={r['recall']:.3f}{r['recall_ci95']} F1={r['f1']:.3f}{r['f1_ci95']} rec={r['recall_fraction']}")


# ================================================================ S3
# Per-class ECE (10-bin), Brier, reliability points.
def ece_brier(y, prob, nbins=10):
    brier = float(np.mean((prob - y) ** 2))
    bins = np.linspace(0.0, 1.0, nbins + 1)
    ece = 0.0
    points = []
    for b in range(nbins):
        lo, hi = bins[b], bins[b + 1]
        if b == nbins - 1:
            mask = (prob >= lo) & (prob <= hi)
        else:
            mask = (prob >= lo) & (prob < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        conf = float(prob[mask].mean())
        acc = float(y[mask].mean())
        ece += (n / len(prob)) * abs(acc - conf)
        points.append({"bin_lo": float(lo), "bin_hi": float(hi), "n": n,
                       "mean_pred": conf, "empirical_freq": acc})
    return float(ece), brier, points


s3 = {"n_bins": 10, "classes": {}}
pooled_ece_num = 0.0
for c in range(C):
    e, br, pts = ece_brier(labels[:, c], probs[:, c], 10)
    s3["classes"][class_names[c]] = {"ece": e, "brier": br,
                                     "n_pos": int(labels[:, c].sum()),
                                     "reliability_points": pts}
# pooled ECE over all class-cell decisions (flattened)
flat_y = labels.reshape(-1)
flat_p = probs.reshape(-1)
pooled_e, pooled_b, _ = ece_brier(flat_y, flat_p, 10)
s3["pooled_ece_flat"] = pooled_e
s3["pooled_brier_flat"] = pooled_b
json.dump(s3, open(os.path.join(OUT, "per_class_calibration.json"), "w"), indent=2)
print("[S3] written per_class_calibration.json; pooled ECE(flat)=%.4f Brier=%.4f" % (pooled_e, pooled_b))
for cn in class_names:
    print(f"  {cn:16s} ECE={s3['classes'][cn]['ece']:.4f} Brier={s3['classes'][cn]['brier']:.4f} n_pos={s3['classes'][cn]['n_pos']}")

# reliability figure
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
axes = axes.ravel()
for c in range(C):
    ax = axes[c]
    pts = s3["classes"][class_names[c]]["reliability_points"]
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.6)
    if pts:
        mp = [q["mean_pred"] for q in pts]
        ef = [q["empirical_freq"] for q in pts]
        ns = [q["n"] for q in pts]
        ax.plot(mp, ef, "-o", ms=4, color="C0")
        for x, y, n in zip(mp, ef, ns):
            ax.annotate(str(n), (x, y), fontsize=6, alpha=0.7,
                        xytext=(2, 2), textcoords="offset points")
    ax.set_title(f"{class_names[c]}\nECE={s3['classes'][class_names[c]]['ece']:.3f} "
                 f"Brier={s3['classes'][class_names[c]]['brier']:.3f}", fontsize=9)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("mean predicted prob", fontsize=7)
    ax.set_ylabel("empirical freq", fontsize=7)
    ax.tick_params(labelsize=6)
axes[7].axis("off")
axes[7].text(0.1, 0.5, f"Pooled (flat)\nECE={pooled_e:.3f}\nBrier={pooled_b:.3f}\nn_test={N}",
             fontsize=10, va="center")
fig.suptitle("Per-class reliability diagrams (10-bin), bin counts annotated", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(OUT, "per_class_calibration.pdf"))
plt.close(fig)


# ================================================================ S4
# Clinical utility.
# Binary abnormal-vs-normal: positive = ANY abnormality present.
# Normal class = index 0. "Any abnormality" = any of classes 1..6 labeled 1.
abn_label = (labels[:, 1:].sum(1) > 0).astype(np.int64)         # ground truth abnormal
# score for abnormality = max prob over abnormal classes
abn_score = probs[:, 1:].max(1)
n_pos = int(abn_label.sum()); n_neg = int(N - abn_label.sum())
prev = n_pos / N

# (a) Decision curve / net benefit
pts_thr = np.linspace(0.01, 0.99, 99)
nb_model, nb_all = [], []
for pt in pts_thr:
    flagged = (abn_score >= pt).astype(np.int64)
    tp = int(((flagged == 1) & (abn_label == 1)).sum())
    fp = int(((flagged == 1) & (abn_label == 0)).sum())
    nb = tp / N - fp / N * (pt / (1 - pt))
    nb_model.append(nb)
    # treat-all
    tp_a = n_pos; fp_a = n_neg
    nb_all.append(tp_a / N - fp_a / N * (pt / (1 - pt)))
nb_none = [0.0] * len(pts_thr)

# AUROC for abnormality (rank-based, tie-corrected)
def auroc(y, s):
    order = np.argsort(s)
    ranks = np.empty(len(s), float)
    sr = s[order]
    i = 0
    while i < len(sr):
        j = i
        while j + 1 < len(sr) and sr[j + 1] == sr[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    pos = y == 1
    n1 = int(pos.sum()); n0 = int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))

abn_auroc = auroc(abn_label, abn_score)

# (b) Triage / deferral curve using multi-label per-cell correctness.
# A cell is "correct" if its thresholded multi-label prediction == labels exactly (subset accuracy).
cell_correct = (preds == labels).all(1).astype(np.int64)
# confidence = margin of the multi-label decision: min over classes of |prob - threshold|
cell_conf = np.min(np.abs(probs - thresholds[None, :]), axis=1)
order_conf = np.argsort(-cell_conf)          # most confident first
sorted_correct = cell_correct[order_conf]
retained_frac, retained_acc, deferred_frac = [], [], []
for k in range(1, N + 1):
    retained_frac.append(k / N)
    retained_acc.append(float(sorted_correct[:k].mean()))
    deferred_frac.append(1 - k / N)
base_subset_acc = float(cell_correct.mean())

s4 = {
    "abnormal_vs_normal": {
        "positive_definition": "any of classes 1..6 labeled 1",
        "n_pos": n_pos, "n_neg": n_neg, "prevalence": prev,
        "auroc": abn_auroc,
        "decision_curve": {
            "threshold_prob": pts_thr.tolist(),
            "net_benefit_model": nb_model,
            "net_benefit_treat_all": nb_all,
            "net_benefit_treat_none": nb_none,
        },
    },
    "triage_yield": {
        "confidence_metric": "min_c |prob_c - threshold_c| (multi-label decision margin)",
        "correctness_metric": "exact multi-label subset accuracy",
        "base_subset_accuracy": base_subset_acc,
        "retained_fraction": retained_frac,
        "retained_accuracy": retained_acc,
        "deferred_fraction": deferred_frac,
    },
}
json.dump(s4, open(os.path.join(OUT, "clinical_utility.json"), "w"), indent=2)
print("[S4] written clinical_utility.json; abn prevalence=%.3f AUROC=%.3f base subset acc=%.3f" %
      (prev, abn_auroc, base_subset_acc))

# decision curve fig
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(pts_thr, nb_model, "-", color="C0", lw=2, label="Model (abnormal-vs-normal)")
ax.plot(pts_thr, nb_all, "-", color="C1", lw=1.2, label="Treat all")
ax.plot(pts_thr, nb_none, "-", color="gray", lw=1.0, label="Treat none")
ax.axhline(0, color="k", lw=0.5)
ax.set_ylim(min(-0.05, prev * -0.3), prev * 1.1 + 0.02)
ax.set_xlabel("Threshold probability p_t")
ax.set_ylabel("Net benefit")
ax.set_title(f"Decision curve: abnormal-vs-normal triage (prev={prev:.2f}, AUROC={abn_auroc:.3f})")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "decision_curve.pdf")); plt.close(fig)

# triage yield fig
fig, ax1 = plt.subplots(figsize=(7, 5))
ax1.plot(np.array(deferred_frac), retained_acc, "-", color="C0", lw=2)
ax1.axhline(base_subset_acc, color="C0", ls="--", lw=0.8, alpha=0.6,
            label=f"no deferral acc={base_subset_acc:.3f}")
ax1.set_xlabel("Fraction of cells deferred (least confident flagged)")
ax1.set_ylabel("Subset accuracy on retained cells", color="C0")
ax1.tick_params(axis="y", labelcolor="C0")
ax1.set_ylim(base_subset_acc - 0.02, 1.005)
ax1.grid(alpha=0.3); ax1.legend(loc="lower right")
ax1.set_title("Triage / deferral curve")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "triage_yield.pdf")); plt.close(fig)
print("triage acc at 10%% defer:", retained_acc[int(0.9 * N) - 1], "at 20%% defer:", retained_acc[int(0.8 * N) - 1])


# ================================================================ M4b
# Slide-leakage audit from filenames.
import re
from collections import defaultdict

def parse(fn):
    m = re.match(r"^([A-Za-z]+)_(\d+)\.", fn)
    if m:
        return m.group(1), m.group(2)
    return None, None

def prefixes(numeric, prefix, k):
    # candidate slide token = species-prefix + first k digits of numeric id
    return f"{prefix}_{numeric[:k]}"

train_parsed = [parse(f) for f in train_files]
val_parsed = [parse(f) for f in val_files]
test_parsed = [parse(f) for f in test_files]

species_counts = defaultdict(int)
digit_len_counts = defaultdict(int)
for pfx, num in train_parsed + val_parsed + test_parsed:
    if pfx is not None:
        species_counts[pfx] += 1
        digit_len_counts[len(num)] += 1

# Audit shared groups between train and test for several k
audit = {"filename_format_example": test_files[:5],
         "species_prefix_counts": dict(species_counts),
         "numeric_id_length_distribution": dict(sorted(digit_len_counts.items())),
         "grouping_by_leading_digits": {}}

for k in [1, 2, 3, 4]:
    train_groups = set(prefixes(n, p, k) for p, n in train_parsed if n)
    test_groups_list = [prefixes(n, p, k) for p, n in test_parsed if n]
    n_test_shared = sum(1 for g in test_groups_list if g in train_groups)
    n_train_groups = len(train_groups)
    n_test_groups = len(set(test_groups_list))
    audit["grouping_by_leading_digits"][f"k={k}"] = {
        "n_train_groups": n_train_groups,
        "n_test_groups": n_test_groups,
        "n_test_cells_sharing_group_with_train": int(n_test_shared),
        "frac_test_sharing": round(n_test_shared / len(test_parsed), 4),
    }

# Exact whole-id overlap (identical cell in train and test?)
train_ids = set((p, n) for p, n in train_parsed)
val_ids = set((p, n) for p, n in val_parsed)
test_ids = set((p, n) for p, n in test_parsed)
audit["exact_id_overlap"] = {
    "train_test_identical_ids": len(train_ids & test_ids),
    "val_test_identical_ids": len(val_ids & test_ids),
    "train_val_identical_ids": len(train_ids & val_ids),
}
# Consecutive-id clustering hint: are ids globally sequential (single acquisition run)?
all_nums = sorted(int(n) for p, n in (train_parsed + val_parsed + test_parsed) if n)
audit["numeric_id_range"] = {"min": all_nums[0], "max": all_nums[-1], "n_unique": len(set(all_nums))}
json.dump(audit, open(os.path.join(OUT, "_slide_leakage_audit.json"), "w"), indent=2)
print("[M4b] audit computed")
print(json.dumps(audit["grouping_by_leading_digits"], indent=2))
print("exact overlap:", audit["exact_id_overlap"])
print("species:", dict(species_counts))
