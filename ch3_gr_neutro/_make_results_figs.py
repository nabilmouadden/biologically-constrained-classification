"""Build clean clinical results figures 1-3 (and 4 from reliability JSON when present).
NO EM DASHES anywhere. Plain clinical labels."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

FIGS = Path("/home/nabil/Téléchargements/biologically-constrained-classification-main/papers/biology_journal/figs")
FIGS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12.5,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
    "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#4a4a4a", "axes.linewidth": 0.8, "figure.dpi": 150,
})

INK = "#1a1a1a"; MUTED = "#5a5a5a"; GRID = "#d9d9d9"
BAR = "#2c7fb8"          # single-series clinical blue
# progression light -> dark blue (Original, Re-annotated, + stain-robust)
SEQ3 = ["#a6cee3", "#4292c6", "#08519c"]


def save(fig, name):
    fig.savefig(FIGS / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGS / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


# ---------------- Figure 1: per-category F1 (internal) ----------------
# 5-seed tuned means +/- sd, R1_RESULTS_FINAL.md section 4
cats = ["Normal", "Hypogranulation", "Hyposegmentation", "Hypergranulation",
        "Chromatin hypercondensation", "Hypersegmentation", "Döhle bodies"]
f1   = [0.965, 0.931, 0.887, 0.859, 0.806, 0.721, 0.705]
sd   = [0.012, 0.025, 0.016, 0.066, 0.052, 0.060, 0.045]

fig, ax = plt.subplots(figsize=(7.4, 4.3))
ypos = np.arange(len(cats))[::-1]  # strongest at top
ax.barh(ypos, f1, height=0.62, color=BAR, edgecolor="white", linewidth=0.8,
        xerr=sd, error_kw=dict(ecolor=MUTED, elinewidth=1.1, capsize=3, capthick=1.1))
ax.set_yticks(ypos); ax.set_yticklabels(cats)
ax.set_xlim(0, 1.0)
ax.set_xlabel("F1 score (mean of five seeds)")
ax.set_title("Per-category detection in the internal test set", pad=10, color=INK)
ax.xaxis.set_major_formatter(lambda v, _: f"{v:.1f}")
ax.grid(axis="x", color=GRID, linewidth=0.7, zorder=0)
ax.set_axisbelow(True)
for yp, v, e in zip(ypos, f1, sd):
    ax.text(v + e + 0.02, yp, f"{v:.2f}", va="center", ha="left",
            fontsize=10.5, color=INK)
save(fig, "fig_per_class")


# ---------------- Figure 2: review-referral (deferral) curve ----------------
pct = np.array([0, 5, 10, 15, 20])
acc = np.array([0.813, 0.839, 0.863, 0.887, 0.910])
asd = np.array([0.022, 0.018, 0.016, 0.014, 0.012])

fig, ax = plt.subplots(figsize=(7.0, 4.3))
ax.fill_between(pct, acc - asd, acc + asd, color=BAR, alpha=0.15, linewidth=0)
ax.plot(pct, acc, "-o", color=BAR, markersize=8, linewidth=2.2,
        markeredgecolor="white", markeredgewidth=1.2, zorder=3)
for x, y in zip(pct, acc):
    ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 11),
                ha="center", fontsize=10.5, color=INK)
ax.set_xticks(pct)
ax.xaxis.set_major_formatter(PercentFormatter())
ax.set_xlim(-1.5, 21.5); ax.set_ylim(0.79, 0.935)
ax.set_xlabel("Least confident cells referred for expert review")
ax.set_ylabel("All labels correct, on retained cells")
ax.set_title("Accuracy on retained cells as uncertain cells are referred for review",
             pad=10, color=INK, fontsize=13)
ax.grid(color=GRID, linewidth=0.7); ax.set_axisbelow(True)
save(fig, "fig_referral")


# ---------------- Figure 3: external transfer grouped bars ----------------
metrics = ["AUROC", "Specificity", "Hypogranulation\nrecall", "Döhle\nrecall"]
models = ["Original", "Corrected labels", "Corrected labels, stain robust"]
vals = np.array([
    [0.851, 0.440, 0.131, 0.109],   # Original
    [0.862, 0.537, 0.663, 0.314],   # Re-annotated
    [0.906, 0.775, 0.704, 0.318],   # + stain robust
])
# 5-seed sd available only for the final (stain-robust) model
err_final = [0.013, 0.072, 0.080, 0.114]

fig, ax = plt.subplots(figsize=(8.0, 4.9))
x = np.arange(len(metrics)); w = 0.26
for i, (m, c) in enumerate(zip(models, SEQ3)):
    off = (i - 1) * w
    err = err_final if i == 2 else None
    bars = ax.bar(x + off, vals[i], w, color=c, edgecolor="white", linewidth=0.8,
                  label=m, yerr=err,
                  error_kw=dict(ecolor=MUTED, elinewidth=1.0, capsize=2.5, capthick=1.0))
    for j, (b, v) in enumerate(zip(bars, vals[i])):
        top = v + (err_final[j] if i == 2 else 0) + 0.015
        ax.text(b.get_x() + b.get_width() / 2, top, f"{v:.2f}",
                ha="center", va="bottom", fontsize=8.6, color=INK)
ax.set_xticks(x); ax.set_xticklabels(metrics)
ax.set_ylim(0, 1.06)
ax.set_ylabel("Score on the external dataset")
ax.yaxis.set_major_formatter(lambda v, _: f"{v:.1f}")
ax.grid(axis="y", color=GRID, linewidth=0.7); ax.set_axisbelow(True)
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=3,
          columnspacing=1.4, handlelength=1.3)
ax.set_title("External transfer to the Barrera-Merino dataset",
             pad=42, color=INK)
ax.text(0.5, -0.24, "Error bars: standard deviation across five training runs (final model).",
        transform=ax.transAxes, ha="center", fontsize=9, color=MUTED)
save(fig, "fig_external")

print("figs 1-3 done")


# ---------------- Figure 4: reliability diagram (calibration) ----------------
# Real per-cell probabilities pooled over 5 seeds x 7 per-category binaries,
# reannotated model. Numbers computed on Ruche from predictions.pt.
rel = json.loads((Path(__file__).parent / "reliability_pooled_5seed.json").read_text())
bins = [b for b in rel["bins_10_equalwidth"] if b["count"] > 0]
xp = np.array([b["mean_pred"] for b in bins])
yo = np.array([b["obs_freq"] for b in bins])
cnt = np.array([b["count"] for b in bins], dtype=float)
ece = rel["ece_15bin_equalwidth"]
brier = rel["pooled_brier"]

fig, ax = plt.subplots(figsize=(6.2, 5.6))
ax.plot([0, 1], [0, 1], ls="--", color="#9a9a9a", linewidth=1.4,
        label="Perfect calibration", zorder=1)
# marker area scaled by sqrt(count)
sizes = 40 + 360 * (np.sqrt(cnt) / np.sqrt(cnt.max()))
ax.plot(xp, yo, "-", color=BAR, linewidth=1.8, zorder=2, alpha=0.9)
ax.scatter(xp, yo, s=sizes, color=BAR, edgecolor="white", linewidth=1.1,
           zorder=3, label="Model, binned by predicted probability")
# annotate counts on the two dominant bins
for x, y, c in zip(xp, yo, cnt):
    if c > 1000:
        ax.annotate(f"n={int(c):,}", (x, y), textcoords="offset points",
                    xytext=(-6, -16) if x < 0.5 else (6, 8),
                    ha="right" if x < 0.5 else "left", fontsize=9, color=MUTED)
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Observed frequency of the category")
ax.set_title("Calibration of category probabilities, pooled",
             pad=10, color=INK)
ax.xaxis.set_major_formatter(lambda v, _: f"{v:.1f}")
ax.yaxis.set_major_formatter(lambda v, _: f"{v:.1f}")
ax.grid(color=GRID, linewidth=0.7); ax.set_axisbelow(True)
ax.text(0.04, 0.93,
        f"Expected calibration error {ece:.3f}\nBrier score {brier:.3f}",
        transform=ax.transAxes, fontsize=11, color=INK, va="top",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#f2f6fa",
                  edgecolor="#c9d6e2", linewidth=0.9))
ax.legend(frameon=False, loc="lower right", fontsize=9.5)
ax.text(0.5, -0.15,
        "Pooled over five training runs and the seven categories; marker size grows with the number of cells in each bin.",
        transform=ax.transAxes, ha="center", fontsize=8.4, color=MUTED)
save(fig, "fig_calibration")
print("fig 4 done")

