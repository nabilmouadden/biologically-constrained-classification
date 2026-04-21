"""Generate all paper figures from completed training runs.

Expects these output dirs under /gpfs/workdir/mouaddenn/thesis/aml_matek/outputs/:
  {backbone}_aml_matek_baseline/               (linear-probe; predictions.pt has is_baseline=True)
  {backbone}_aml_matek_joint_const/
  {backbone}_aml_matek_joint_unconst/
  dinobloom_s_aml_matek_frozen_const/
  dinobloom_s_aml_matek_frozen_unconst/

Produces figures under /gpfs/workdir/mouaddenn/thesis/aml_matek/figures/.
"""

import csv
import json
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                              roc_auc_score)

sys.path.insert(0, str(Path(__file__).parent))
from models import build_prior_C

WORKDIR = Path("/gpfs/workdir/mouaddenn")
THESIS = WORKDIR / "thesis" / "aml_matek"
OUTPUTS = THESIS / "outputs"
FIGDIR = THESIS / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

BACKBONES = ["dinobloom_s", "dinov2_vitb14", "resnet50"]
BACKBONE_DISPLAY = {"dinobloom_s": "DinoBloom-S",
                    "dinov2_vitb14": "DINOv2-ViT-B/14",
                    "resnet50": "ResNet-50"}

CLASS_NAMES = [
    "myeloblast", "promyelocyte", "myelocyte", "metamyelocyte",
    "band", "segmented", "eosinophil", "basophil", "monocyte",
    "lymphocyte", "plasma", "erythroblast", "rbc/plt",
    "rare/atyp", "artifact",
]

# Indices of classes that actually appear in Matek (plasma, rbc_platelet don't).
# Computed dynamically from test labels; this is just a default for display.

# Concept category -> color map, per user spec.
CONCEPT_CATEGORY = {
    "round_oval_nucleus": "nuclear_shape",
    "indented_nucleus": "nuclear_shape",
    "band_nucleus": "nuclear_shape",
    "multilobed_nucleus": "nuclear_shape",
    "eccentric_nucleus": "nuclear_shape",
    "fine_chromatin": "chromatin",
    "condensed_chromatin": "chromatin",
    "visible_nucleoli": "chromatin",
    "high_nc_ratio": "other",
    "low_nc_ratio": "other",
    "large_cell": "other",
    "azurophilic_granules": "granules",
    "neutrophilic_granules": "granules",
    "eosinophilic_granules": "granules",
    "basophilic_granules": "granules",
    "agranular_cytoplasm": "cytoplasm",
    "basophilic_cytoplasm": "cytoplasm",
    "abundant_pale_cytoplasm": "cytoplasm",
}
CATEGORY_COLOR = {
    "nuclear_shape": "#4c72b0",   # blue
    "chromatin":    "#55a868",    # green
    "granules":     "#dd8452",    # orange
    "cytoplasm":    "#c44e52",    # red
    "other":        "#8c8c8c",    # gray
}

ALPHAS = (0.01, 0.05, 0.10)


def load_concept_cfg():
    return json.loads((THESIS / "concept_config.json").read_text())


def load_run(name):
    d = OUTPUTS / name
    if not (d / "predictions.pt").exists():
        return None
    blob = torch.load(d / "predictions.pt", map_location="cpu", weights_only=False)
    blob["_dir"] = d
    blob["_name"] = name
    return blob


def compute_class_metrics(pred, labels):
    """pred: (N, C) logits; labels: (N,) int."""
    y_pred = pred.argmax(1)
    acc = accuracy_score(labels, y_pred)
    w_f1 = f1_score(labels, y_pred, average="weighted", zero_division=0)
    m_f1 = f1_score(labels, y_pred, average="macro", zero_division=0)
    return acc, w_f1, m_f1


def compute_concept_metrics(concept_pred_sig, concept_tgt_soft):
    """Macro mean F1 @ 0.5 over K concepts."""
    tgt_bin = (concept_tgt_soft >= 0.5).astype(int)
    pred_bin = (concept_pred_sig >= 0.5).astype(int)
    K = tgt_bin.shape[1]
    f1s = []
    for k in range(K):
        if tgt_bin[:, k].sum() == 0 or tgt_bin[:, k].sum() == len(tgt_bin):
            continue
        f1s.append(f1_score(tgt_bin[:, k], pred_bin[:, k], zero_division=0))
    return float(np.mean(f1s)) if f1s else float("nan"), f1s


def violation_rate(concept_pred_sig, exclusive_pairs):
    hard = concept_pred_sig >= 0.5
    n = hard.shape[0]
    total = 0
    per_pair = []
    for a, b in exclusive_pairs:
        v = int((hard[:, a] & hard[:, b]).sum())
        total += v
        per_pair.append((a, b, v, v / n))
    rate = total / (n * max(1, len(exclusive_pairs)))
    return rate, per_pair


def completeness_probe(cal_pred_sig, cal_labels, test_pred_sig, test_labels):
    clf = LogisticRegression(max_iter=2000, n_jobs=-1)
    clf.fit(cal_pred_sig, cal_labels)
    return float(clf.score(test_pred_sig, test_labels))


# ---------------------------------------------------------------------------
# FIGURE 1: main results table
# ---------------------------------------------------------------------------
def figure_main_table(cfg):
    concepts = cfg["concepts"]
    cidx = {c: i for i, c in enumerate(concepts)}
    exclusive_pairs = [(cidx[p["concepts"][0]], cidx[p["concepts"][1]])
                       for p in cfg["concept_constraint_matrix"]["mutually_exclusive_pairs"]]

    rows = []
    for bb in BACKBONES:
        configs = [f"{bb}_aml_matek_baseline",
                   f"{bb}_aml_matek_joint_unconst",
                   f"{bb}_aml_matek_joint_const",
                   f"{bb}_aml_matek_joint_const_posw"]
        if bb == "dinobloom_s":
            configs.extend([f"{bb}_aml_matek_frozen_unconst",
                            f"{bb}_aml_matek_frozen_const"])
        for cfg_name in configs:
            r = load_run(cfg_name)
            if r is None:
                continue
            test = r["test"]
            labels = test["labels"].numpy()
            cls_logits = test["class_logits"].numpy()
            acc, w_f1, m_f1 = compute_class_metrics(cls_logits, labels)

            row = dict(
                backbone=BACKBONE_DISPLAY.get(bb, bb),
                config=cfg_name.replace(f"{bb}_aml_matek_", ""),
                class_acc=acc,
                weighted_f1=w_f1,
                macro_f1=m_f1,
                concept_f1=float("nan"),
                violation_rate=float("nan"),
                probe_acc=float("nan"),
                conformal_cov_a05=float("nan"),
            )
            if not r.get("is_baseline", False) and "concept_logits" in test:
                cp = test["concept_logits"].sigmoid().numpy()
                ct_soft = test["concept_targets"].numpy()
                ct = (ct_soft >= 0.5).astype(int)
                row["concept_f1"], _ = compute_concept_metrics(cp, ct_soft)
                row["violation_rate"], _ = violation_rate(cp, exclusive_pairs)
                # completeness probe
                cal = r["cal"]
                cal_cp = cal["concept_logits"].sigmoid().numpy()
                cal_y = cal["labels"].numpy()
                cal_ct = (cal["concept_targets"].numpy() >= 0.5).astype(int)
                row["probe_acc"] = completeness_probe(cal_cp, cal_y, cp, labels)
                # Conformal coverage at α=0.05 (mean over concepts)
                covs = []
                for k in range(cp.shape[1]):
                    s = np.where(cal_ct[:, k] == 1, 1 - cal_cp[:, k], cal_cp[:, k])
                    n_cal = len(s)
                    q = np.quantile(s, min(1.0, math.ceil(0.95 * (n_cal + 1)) / n_cal),
                                    method="higher")
                    detected = cp[:, k] > 1 - q
                    absent = cp[:, k] < q
                    uncertain = ~(detected | absent)
                    covered = ((ct[:, k] == 1) & detected) | ((ct[:, k] == 0) & absent) | uncertain
                    covs.append(float(covered.mean()))
                row["conformal_cov_a05"] = float(np.mean(covs))
            rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = FIGDIR / "main_results_table.csv"
    df.to_csv(csv_path, index=False, float_format="%.4f")

    # LaTeX (incl. Cov@0.05 column)
    tex_path = FIGDIR / "main_results_table.tex"
    with open(tex_path, "w") as f:
        f.write("% Main results table. Generated by make_figures.py.\n")
        f.write("\\begin{tabular}{llccccccc}\n\\toprule\n")
        f.write("Backbone & Config & Acc. & W-F1 & Macro-F1 & Concept F1 & Viol. rate & Probe acc. & Cov$_{\\alpha=.05}$ \\\\\n")
        f.write("\\midrule\n")
        for _, r in df.iterrows():
            def fmt(v, d=3):
                return "-" if (v is None or (isinstance(v, float) and math.isnan(v))) else f"{v:.{d}f}"
            f.write(f"{r['backbone']} & {r['config']} & "
                    f"{fmt(r['class_acc'])} & {fmt(r['weighted_f1'])} & {fmt(r['macro_f1'])} & "
                    f"{fmt(r['concept_f1'])} & {fmt(r['violation_rate'], 4)} & {fmt(r['probe_acc'])} & "
                    f"{fmt(r['conformal_cov_a05'])} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    print(f"[fig1] main results: {len(df)} rows -> {csv_path}")
    return df


# ---------------------------------------------------------------------------
# FIGURE 2: per-concept F1 bar chart (horizontal, sorted, color-coded)
# ---------------------------------------------------------------------------
def figure_per_concept_f1(best_config="dinobloom_s_aml_matek_joint_const"):
    r = load_run(best_config)
    if r is None:
        print(f"[fig2] SKIP: no run {best_config}")
        return
    cfg = load_concept_cfg()
    concepts = cfg["concepts"]
    cp = r["test"]["concept_logits"].sigmoid().numpy()
    ct = r["test"]["concept_targets"].numpy()
    tgt_bin = (ct >= 0.5).astype(int)
    pred_bin = (cp >= 0.5).astype(int)

    f1s = []
    for k in range(len(concepts)):
        if tgt_bin[:, k].sum() == 0 or tgt_bin[:, k].sum() == len(tgt_bin):
            f1s.append(float("nan"))
        else:
            f1s.append(f1_score(tgt_bin[:, k], pred_bin[:, k], zero_division=0))

    rows = sorted(zip(concepts, f1s), key=lambda x: -(x[1] if not math.isnan(x[1]) else -1))
    names = [r[0] for r in rows]
    vals = [0.0 if math.isnan(r[1]) else r[1] for r in rows]
    colors = [CATEGORY_COLOR[CONCEPT_CATEGORY[n]] for n in names]

    fig, ax = plt.subplots(figsize=(7, 6.5))
    y = np.arange(len(names))
    ax.barh(y, vals, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1 @ 0.5")
    ax.set_title(f"Per-concept F1 — {BACKBONE_DISPLAY['dinobloom_s']} (joint + constrained)")
    ax.grid(axis="x", alpha=0.3)

    legend_items = [Patch(color=c, label=k.replace("_", " "))
                    for k, c in CATEGORY_COLOR.items()]
    ax.legend(handles=legend_items, loc="lower right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIGDIR / "per_concept_f1_bar.pdf")
    fig.savefig(FIGDIR / "per_concept_f1_bar.png", dpi=140)
    plt.close(fig)
    print(f"[fig2] per-concept F1 bar -> per_concept_f1_bar.pdf")


# ---------------------------------------------------------------------------
# FIGURE 3: attention map grid (6 cells x 5 cols: image + 4 concepts)
# ---------------------------------------------------------------------------
def figure_attention_grid(best_config="dinobloom_s_aml_matek_joint_const"):
    att_path = OUTPUTS / best_config / "attention_weights.pt"
    if not att_path.exists():
        print(f"[fig3] SKIP: no {att_path}")
        return
    cfg = load_concept_cfg()
    concepts = cfg["concepts"]

    blob = torch.load(att_path, map_location="cpu", weights_only=False)
    attn = blob["attn"].numpy()             # (n, K, P)
    filenames = blob["filenames"]
    labels = blob["labels"].numpy()
    H, W = blob["grid_hw"]

    # Pick one cell per target class. Target classes: myeloblast(0), promyelocyte(1),
    # segmented(5), eosinophil(6), monocyte(8), lymphocyte(9).
    targets = [0, 1, 5, 6, 8, 9]
    row_picks = []
    for tcls in targets:
        where = np.where(labels == tcls)[0]
        if len(where) == 0:
            continue
        row_picks.append(int(where[0]))

    # 4 concepts to show
    concept_picks = ["round_oval_nucleus", "fine_chromatin",
                     "condensed_chromatin", "neutrophilic_granules"]
    cidx = [concepts.index(c) for c in concept_picks]

    n_rows = len(row_picks)
    n_cols = 1 + len(concept_picks)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.2, n_rows * 2.3))
    axes = np.atleast_2d(axes)

    for ri, img_i in enumerate(row_picks):
        try:
            img = Image.open(filenames[img_i]).convert("RGB").resize((224, 224))
        except Exception:
            img = Image.new("RGB", (224, 224), (128, 128, 128))
        ax0 = axes[ri, 0]
        ax0.imshow(img)
        cls_name = CLASS_NAMES[labels[img_i]]
        ax0.set_ylabel(cls_name, fontsize=10)
        ax0.set_xticks([]); ax0.set_yticks([])
        if ri == 0:
            ax0.set_title("image", fontsize=10)
        for ci, k in enumerate(cidx):
            a = attn[img_i, k].reshape(H, W)
            a = (a - a.min()) / (a.max() - a.min() + 1e-9)
            a_up = np.array(Image.fromarray((a * 255).astype(np.uint8))
                            .resize((224, 224), Image.BILINEAR))
            ax = axes[ri, ci + 1]
            ax.imshow(img)
            ax.imshow(a_up, cmap="jet", alpha=0.45)
            ax.set_xticks([]); ax.set_yticks([])
            if ri == 0:
                ax.set_title(concept_picks[ci].replace("_", " "), fontsize=9)

    fig.tight_layout()
    fig.savefig(FIGDIR / "attention_grid.pdf")
    fig.savefig(FIGDIR / "attention_grid.png", dpi=150)
    plt.close(fig)
    print(f"[fig3] attention grid {n_rows}x{n_cols} -> attention_grid.pdf")


# ---------------------------------------------------------------------------
# FIGURE 4: constrained vs unconstrained violation comparison
# ---------------------------------------------------------------------------
def figure_violation_comparison():
    cfg = load_concept_cfg()
    concepts = cfg["concepts"]
    cidx = {c: i for i, c in enumerate(concepts)}
    pairs = cfg["concept_constraint_matrix"]["mutually_exclusive_pairs"]

    # Use DinoBloom joint runs for the comparison.
    r_const = load_run("dinobloom_s_aml_matek_joint_const")
    r_unconst = load_run("dinobloom_s_aml_matek_joint_unconst")
    if r_const is None or r_unconst is None:
        print("[fig4] SKIP: need both joint_const and joint_unconst")
        return

    def rates(blob):
        p = blob["test"]["concept_logits"].sigmoid().numpy()
        hard = p >= 0.5
        n = hard.shape[0]
        out = []
        for pair in pairs:
            a, b = cidx[pair["concepts"][0]], cidx[pair["concepts"][1]]
            out.append(float((hard[:, a] & hard[:, b]).mean()))
        return out

    const_r = rates(r_const)
    unconst_r = rates(r_unconst)

    pair_names = [f"{p['concepts'][0].replace('_', ' ')}\n{p['concepts'][1].replace('_', ' ')}"
                  for p in pairs]

    order = np.argsort(-np.array(unconst_r))  # sort by unconstrained rate (descending)
    pair_names = [pair_names[i] for i in order]
    const_r = [const_r[i] for i in order]
    unconst_r = [unconst_r[i] for i in order]

    x = np.arange(len(pair_names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - width/2, unconst_r, width, label="unconstrained", color="#c44e52",
           edgecolor="black", linewidth=0.4)
    ax.bar(x + width/2, const_r, width, label="constrained", color="#4c72b0",
           edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(pair_names, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Violation rate")
    ax.set_title("Mutually exclusive pair violation rate (DinoBloom-S, joint)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGDIR / "violation_comparison.pdf")
    fig.savefig(FIGDIR / "violation_comparison.png", dpi=140)
    plt.close(fig)
    print(f"[fig4] violation comparison -> violation_comparison.pdf "
          f"(total: unconst={np.mean(unconst_r):.4f}, const={np.mean(const_r):.4f})")


# ---------------------------------------------------------------------------
# FIGURE 5: learned co-occurrence vs prior C (side by side)
# ---------------------------------------------------------------------------
def figure_cooccur_vs_prior(best_config="dinobloom_s_aml_matek_joint_const"):
    r = load_run(best_config)
    if r is None:
        print(f"[fig5] SKIP: no {best_config}")
        return
    cfg = load_concept_cfg()
    concepts = cfg["concepts"]
    prior_C = build_prior_C(concepts, cfg["concept_constraint_matrix"]).numpy()

    cp = r["test"]["concept_logits"].sigmoid().numpy()
    # Pearson correlation across test samples
    mean = cp.mean(0, keepdims=True)
    centered = cp - mean
    cov = centered.T @ centered
    std = np.sqrt(np.diag(cov))
    denom = np.outer(std, std)
    denom[denom < 1e-9] = 1e-9
    corr = cov / denom

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, M, title in [(axes[0], prior_C, "Prior constraint matrix  C"),
                          (axes[1], corr, "Learned co-occurrence (test-set corr.)")]:
        sns.heatmap(M, ax=ax, xticklabels=concepts, yticklabels=concepts,
                    cmap="coolwarm", center=0, vmin=-1, vmax=1, square=True,
                    cbar_kws={"shrink": 0.7})
        ax.set_title(title, fontsize=11)
        ax.tick_params(axis="x", rotation=60, labelsize=7)
        ax.tick_params(axis="y", rotation=0, labelsize=7)
        plt.setp(ax.get_xticklabels(), ha="right")
    fig.tight_layout()
    fig.savefig(FIGDIR / "cooccur_vs_prior.pdf")
    fig.savefig(FIGDIR / "cooccur_vs_prior.png", dpi=130)
    plt.close(fig)
    print(f"[fig5] co-occurrence vs prior -> cooccur_vs_prior.pdf")


# ---------------------------------------------------------------------------
# FIGURE 6: conformal coverage — empirical vs nominal
# ---------------------------------------------------------------------------
def figure_conformal_coverage(best_config="dinobloom_s_aml_matek_joint_const"):
    r = load_run(best_config)
    if r is None:
        print(f"[fig6] SKIP: no {best_config}")
        return
    cfg = load_concept_cfg()
    concepts = cfg["concepts"]

    cp_cal = r["cal"]["concept_logits"].sigmoid().numpy()
    ct_cal = (r["cal"]["concept_targets"].numpy() >= 0.5).astype(int)
    cp_test = r["test"]["concept_logits"].sigmoid().numpy()
    ct_test = (r["test"]["concept_targets"].numpy() >= 0.5).astype(int)
    K = cp_cal.shape[1]

    rows = []
    for alpha in ALPHAS:
        per_concept_cov = []
        per_concept_size = []
        for k in range(K):
            pc, tc = cp_cal[:, k], ct_cal[:, k]
            s = np.where(tc == 1, 1 - pc, pc)
            n = len(s)
            q = np.quantile(s, min(1.0, math.ceil((1 - alpha) * (n + 1)) / n),
                            method="higher")
            pt, tt = cp_test[:, k], ct_test[:, k]
            detected = pt > 1 - q
            absent = pt < q
            uncertain = ~(detected | absent)
            covered = ((tt == 1) & detected) | ((tt == 0) & absent) | uncertain
            per_concept_cov.append(float(covered.mean()))
            # set size: 2 if uncertain, 1 otherwise
            per_concept_size.append(float(1.0 + uncertain.mean()))
        rows.append((alpha, per_concept_cov, per_concept_size))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    # LEFT: scatter per concept per alpha
    markers = {0.01: "o", 0.05: "s", 0.10: "^"}
    colors = {0.01: "#c44e52", 0.05: "#4c72b0", 0.10: "#55a868"}
    for alpha, cov, size in rows:
        nominal = 1 - alpha
        axes[0].scatter([nominal] * K, cov, marker=markers[alpha],
                        color=colors[alpha], s=35, alpha=0.75,
                        label=f"α={alpha} (nom {nominal:.2f})", edgecolor="black",
                        linewidth=0.3)
    axes[0].plot([0.8, 1.0], [0.8, 1.0], color="black", linestyle="--", linewidth=1)
    axes[0].set_xlim(0.85, 1.005)
    axes[0].set_ylim(0.8, 1.02)
    axes[0].set_xlabel("Nominal coverage  1-α")
    axes[0].set_ylabel("Empirical coverage per concept")
    axes[0].set_title("Conformal coverage — empirical vs nominal")
    axes[0].legend(loc="lower right", fontsize=8)
    axes[0].grid(alpha=0.3)
    # RIGHT: avg set size per alpha
    alphas_arr = [r[0] for r in rows]
    mean_size = [float(np.mean(r[2])) for r in rows]
    mean_cov = [float(np.mean(r[1])) for r in rows]
    axes[1].bar([f"α={a}" for a in alphas_arr], mean_size, color="#4c72b0",
                edgecolor="black", linewidth=0.4)
    axes[1].set_ylabel("Mean prediction set size")
    axes[1].set_title("Prediction set size")
    axes[1].grid(axis="y", alpha=0.3)
    for i, v in enumerate(mean_size):
        axes[1].text(i, v + 0.01, f"{v:.2f}\ncov {mean_cov[i]:.3f}",
                     ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(FIGDIR / "conformal_coverage.pdf")
    fig.savefig(FIGDIR / "conformal_coverage.png", dpi=140)
    plt.close(fig)
    print(f"[fig6] conformal -> conformal_coverage.pdf "
          f"(coverage at α=0.05: {mean_cov[1]:.3f})")


# ---------------------------------------------------------------------------
# FIGURE 7: training curves
# ---------------------------------------------------------------------------
def figure_training_curves(best_config="dinobloom_s_aml_matek_joint_const"):
    log_path = OUTPUTS / best_config / "training_log.csv"
    if not log_path.exists():
        print(f"[fig7] SKIP: no log for {best_config}")
        return
    df = pd.read_csv(log_path)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    # Loss
    axes[0].plot(df["epoch"], df["train_total"], label="total", color="black")
    axes[0].plot(df["epoch"], df["train_cls"], label="classification", color="#4c72b0")
    axes[0].plot(df["epoch"], df["train_concept"], label="concept (BCE)", color="#55a868")
    # constraint loss on secondary y (it's larger magnitude)
    ax0b = axes[0].twinx()
    ax0b.plot(df["epoch"], df["train_constraint"], label="constraint", color="#c44e52", alpha=0.6)
    ax0b.set_ylabel("constraint loss", color="#c44e52", fontsize=9)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Training losses")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(alpha=0.3)

    # Class accuracy
    axes[1].plot(df["epoch"], df["train_class_acc"], label="train", color="#4c72b0")
    axes[1].plot(df["epoch"], df["val_class_acc"], label="val", color="#c44e52")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].set_title("Classification accuracy")
    axes[1].set_ylim(0, 1.02)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # Concept F1
    axes[2].plot(df["epoch"], df["train_concept_f1_at_0p5"], label="train", color="#4c72b0")
    axes[2].plot(df["epoch"], df["val_concept_f1_at_0p5"], label="val", color="#c44e52")
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("mean concept F1 @ 0.5")
    axes[2].set_title("Concept F1")
    axes[2].set_ylim(0, 1.02)
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGDIR / "training_curves.pdf")
    fig.savefig(FIGDIR / "training_curves.png", dpi=140)
    plt.close(fig)
    print(f"[fig7] training curves -> training_curves.pdf")


# ---------------------------------------------------------------------------
# FIGURE 8: confusion matrix (row-normalized) — classes present in Matek
# ---------------------------------------------------------------------------
def figure_confusion(best_config="dinobloom_s_aml_matek_joint_const"):
    r = load_run(best_config)
    if r is None:
        print(f"[fig8] SKIP: no {best_config}")
        return
    y_true = r["test"]["labels"].numpy()
    y_pred = r["test"]["class_logits"].argmax(1).numpy()

    present = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
    labels_sub = [CLASS_NAMES[i] for i in present]

    cm = confusion_matrix(y_true, y_pred, labels=present)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues",
                xticklabels=labels_sub, yticklabels=labels_sub, ax=ax,
                cbar_kws={"label": "row-normalized prob."}, square=True,
                annot_kws={"size": 8})
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"Confusion matrix — {BACKBONE_DISPLAY['dinobloom_s']} joint+constrained "
                 f"({len(present)}×{len(present)}, classes present in Matek)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGDIR / "confusion_matrix.pdf")
    fig.savefig(FIGDIR / "confusion_matrix.png", dpi=140)
    plt.close(fig)
    print(f"[fig8] confusion matrix {len(present)}x{len(present)} -> confusion_matrix.pdf")


def figure_per_concept_f1_comparison(cfg):
    """Grouped horizontal bars: default BCE (blue) vs posw BCE (orange), one pair per concept.
    Sort by default F1 descending. Annotate band_nucleus recovery.
    DinoBloom-S joint+const, both default and posw."""
    r_def = load_run("dinobloom_s_aml_matek_joint_const")
    r_pw = load_run("dinobloom_s_aml_matek_joint_const_posw")
    if r_def is None or r_pw is None:
        print("[fig_cmp] SKIP: need both default and posw runs")
        return
    concepts = cfg["concepts"]

    def per_concept(r):
        cp = r["test"]["concept_logits"].sigmoid().numpy()
        ct = (r["test"]["concept_targets"].numpy() >= 0.5).astype(int)
        f1s = []
        for k in range(len(concepts)):
            if ct[:, k].sum() == 0 or ct[:, k].sum() == len(ct):
                f1s.append(float("nan"))
            else:
                f1s.append(f1_score(ct[:, k], (cp[:, k] >= 0.5).astype(int), zero_division=0))
        return np.array(f1s)

    f1_def = per_concept(r_def)
    f1_pw = per_concept(r_pw)

    # Sort by default F1 desc (NaNs at end).
    order = np.argsort(-np.where(np.isnan(f1_def), -1, f1_def))
    names = [concepts[i] for i in order]
    d = np.where(np.isnan(f1_def[order]), 0.0, f1_def[order])
    p = np.where(np.isnan(f1_pw[order]), 0.0, f1_pw[order])

    fig, ax = plt.subplots(figsize=(8, 7))
    y = np.arange(len(names))
    h = 0.4
    ax.barh(y - h/2, d, h, color="#4c72b0", edgecolor="black", linewidth=0.4,
            label="default BCE")
    ax.barh(y + h/2, p, h, color="#dd8452", edgecolor="black", linewidth=0.4,
            label="class-weighted BCE (posw)")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1 @ 0.5")
    ax.set_title("Per-concept F1: default vs class-weighted BCE\nDinoBloom-S, joint + constrained")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    # Annotate band_nucleus recovery
    try:
        i_band = names.index("band_nucleus")
        ax.annotate(f"{f1_def[concepts.index('band_nucleus')]:.3f} → {f1_pw[concepts.index('band_nucleus')]:.3f}",
                    xy=(p[i_band], i_band + h/2),
                    xytext=(p[i_band] + 0.1, i_band + h/2),
                    fontsize=9, color="#b23f3f",
                    arrowprops=dict(arrowstyle="->", color="#b23f3f", lw=0.8),
                    verticalalignment="center")
    except ValueError:
        pass

    fig.tight_layout()
    fig.savefig(FIGDIR / "per_concept_f1_comparison.pdf")
    fig.savefig(FIGDIR / "per_concept_f1_comparison.png", dpi=140)
    plt.close(fig)
    print(f"[fig_cmp] per-concept F1 comparison -> per_concept_f1_comparison.pdf")


def figure_lambda_pareto(cfg):
    """Scatter/line: X=total violation rate, Y=class accuracy. One point per λ.
    Reads from outputs/lambda_sweep/lam{X}p{Y}_posw/."""
    concepts = cfg["concepts"]
    cidx = {c: i for i, c in enumerate(concepts)}
    exclusive_pairs = [(cidx[p["concepts"][0]], cidx[p["concepts"][1]])
                       for p in cfg["concept_constraint_matrix"]["mutually_exclusive_pairs"]]
    sweep_dir = OUTPUTS / "lambda_sweep"
    if not sweep_dir.exists():
        print("[fig_pareto] SKIP: no lambda_sweep dir")
        return

    lams, accs, viols, macros, conceptf1s = [], [], [], [], []
    for d in sorted(sweep_dir.iterdir()):
        if not d.is_dir() or not (d / "predictions.pt").exists():
            continue
        # parse lambda from folder name lamXpY_posw
        name = d.name
        try:
            tail = name.replace("lam", "").replace("_posw", "")
            lam = float(tail.replace("p", "."))
        except ValueError:
            print(f"[fig_pareto] skipping unparseable dir: {name}")
            continue
        b = torch.load(d / "predictions.pt", map_location="cpu", weights_only=False)
        test = b["test"]
        y = test["labels"].numpy()
        y_pred = test["class_logits"].argmax(1).numpy()
        cp = test["concept_logits"].sigmoid().numpy()
        ct = (test["concept_targets"].numpy() >= 0.5).astype(int)
        v, _ = violation_rate(cp, exclusive_pairs)
        f1s = [f1_score(ct[:, k], (cp[:, k] >= 0.5).astype(int), zero_division=0)
               for k in range(cp.shape[1]) if 0 < ct[:, k].sum() < len(ct)]
        lams.append(lam)
        accs.append(accuracy_score(y, y_pred))
        viols.append(v)
        macros.append(f1_score(y, y_pred, average="macro", zero_division=0))
        conceptf1s.append(float(np.mean(f1s)))

    if len(lams) < 2:
        print("[fig_pareto] SKIP: not enough points")
        return

    order = np.argsort(lams)
    lams = [lams[i] for i in order]
    accs = [accs[i] for i in order]
    viols = [viols[i] for i in order]
    macros = [macros[i] for i in order]
    conceptf1s = [conceptf1s[i] for i in order]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(viols, accs, marker="o", color="#4c72b0", lw=1.5, markersize=7)
    # Log-x scale spreads the tiny violation-rate values so labels don't overlap.
    ax.set_xscale("log")
    # Pad y-axis so the highest/lowest points have room for offset labels.
    y_span = max(accs) - min(accs)
    y_pad = max(y_span * 1.0, 0.002)
    ax.set_ylim(min(accs) - y_pad, max(accs) + y_pad)
    for lam, v, a in zip(lams, viols, accs):
        # 8px offset to upper-right of each point, stays inside axes.
        ax.annotate(f"λ={lam}", xy=(v, a), xytext=(8, 6),
                    textcoords="offset points",
                    fontsize=9, ha="left", va="bottom")
    ax.set_xlabel("Total violation rate (log scale)")
    ax.set_ylabel("Classification accuracy")
    ax.set_title("Violation–accuracy Pareto curve\n(DinoBloom-S, joint + posw, λ sweep)")
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    ax.plot(lams, macros, marker="s", color="#55a868", lw=1.5, label="macro-F1 (class)")
    ax.plot(lams, conceptf1s, marker="^", color="#dd8452", lw=1.5, label="concept F1")
    ax.set_xlabel("λ (constraint weight)")
    ax.set_ylabel("F1")
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_title("Macro & concept F1 vs λ")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGDIR / "lambda_pareto.pdf")
    fig.savefig(FIGDIR / "lambda_pareto.png", dpi=140)
    plt.close(fig)
    print(f"[fig_pareto] λ sweep -> lambda_pareto.pdf  (n points = {len(lams)})")


def figure_backbone_comparison(cfg):
    """Grouped bars: 3 backbones × 4 metrics (class acc, concept F1, probe acc, 1-violation).
    All from joint_const_posw."""
    concepts = cfg["concepts"]
    cidx = {c: i for i, c in enumerate(concepts)}
    exclusive_pairs = [(cidx[p["concepts"][0]], cidx[p["concepts"][1]])
                       for p in cfg["concept_constraint_matrix"]["mutually_exclusive_pairs"]]

    data = []
    for bb in BACKBONES:
        r = load_run(f"{bb}_aml_matek_joint_const_posw")
        if r is None:
            print(f"[fig_bb] SKIP: no posw run for {bb}")
            return
        test = r["test"]; cal = r["cal"]
        y = test["labels"].numpy()
        y_pred = test["class_logits"].argmax(1).numpy()
        cp = test["concept_logits"].sigmoid().numpy()
        ct = (test["concept_targets"].numpy() >= 0.5).astype(int)
        acc = accuracy_score(y, y_pred)
        f1s = [f1_score(ct[:, k], (cp[:, k] >= 0.5).astype(int), zero_division=0)
               for k in range(cp.shape[1]) if 0 < ct[:, k].sum() < len(ct)]
        cF1 = float(np.mean(f1s))
        v, _ = violation_rate(cp, exclusive_pairs)
        cal_cp = cal["concept_logits"].sigmoid().numpy()
        cal_y = cal["labels"].numpy()
        probe = completeness_probe(cal_cp, cal_y, cp, y)
        data.append((BACKBONE_DISPLAY[bb], acc, cF1, probe, 1.0 - v))

    labels = ["Class acc.", "Concept F1", "Probe acc.", "1 − viol. rate"]
    x = np.arange(len(data))
    w = 0.20
    fig, ax = plt.subplots(figsize=(10, 5))
    offsets = np.linspace(-1.5, 1.5, 4) * w
    colors = ["#4c72b0", "#55a868", "#dd8452", "#c44e52"]
    for i, (lab, col) in enumerate(zip(labels, colors)):
        vals = [d[i + 1] for d in data]
        bars = ax.bar(x + offsets[i], vals, w, label=lab, color=col,
                      edgecolor="black", linewidth=0.4)
        for xi, v in zip(x + offsets[i], vals):
            ax.text(xi, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([d[0] for d in data])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Backbone comparison — joint + constrained + class-weighted BCE")
    ax.legend(loc="lower right", ncol=2, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGDIR / "backbone_comparison.pdf")
    fig.savefig(FIGDIR / "backbone_comparison.png", dpi=140)
    plt.close(fig)
    print(f"[fig_bb] backbone comparison -> backbone_comparison.pdf")


def main():
    cfg = load_concept_cfg()
    print(f"[make_figures] outputs at {OUTPUTS}")
    print(f"[make_figures] figures -> {FIGDIR}")
    df = figure_main_table(cfg)
    figure_per_concept_f1()
    figure_per_concept_f1_comparison(cfg)
    figure_attention_grid()
    figure_violation_comparison()
    figure_cooccur_vs_prior()
    figure_conformal_coverage()
    figure_training_curves()
    figure_confusion()
    figure_lambda_pareto(cfg)
    figure_backbone_comparison(cfg)
    print("\n[done] all figures generated.")


if __name__ == "__main__":
    main()
