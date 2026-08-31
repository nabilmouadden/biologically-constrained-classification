"""Generate all 10 brief-required figures + extra misclassification visuals.

Reads outputs/<tag>/{predictions.pt, eval.json, model.pt} and
outputs/<baseline_tag>/* for the comparison columns.

Outputs go to ch3_gr_neutro/figures/.

Required figures (per Step 6 of the task brief):
  1.  main_results_table.csv  + .tex
  2.  per_concept_f1_bar.pdf
  3.  per_concept_f1_comparison.pdf  (default vs class-weighted BCE)
  4.  violation_comparison.pdf       (constrained vs unconstrained)
  5.  attention_grid.pdf             (one row per cell, columns = concepts)
  6.  conformal_coverage.pdf
  7.  cooccur_vs_prior.pdf
  8.  confusion_matrix.pdf
  9.  training_curves.pdf
  10. lambda_pareto.pdf

Plus user-requested extras:
  E1. misclass_examples.pdf  — misclassified cells with attention over key concepts
  E2. concept_exemplars.pdf  — per concept, top-K cells maximizing that concept
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


HERE = Path(__file__).resolve().parent
WORKDIR = Path("/gpfs/workdir/mouaddenn")


def load_run(tag: str, root: Path):
    rd = root / tag
    pred = torch.load(rd / "predictions.pt", map_location="cpu", weights_only=False)
    summary = json.loads((rd / "summary.json").read_text())
    ev = (rd / "eval.json")
    eval_blob = json.loads(ev.read_text()) if ev.exists() else {}
    log_path = rd / "training_log.csv"
    log = list(csv.reader(open(log_path))) if log_path.exists() else []
    return dict(pred=pred, summary=summary, eval=eval_blob, log=log, run_dir=rd)


# ------------------------- 1. main_results_table ----------------------------
def fig1_main_table(runs: dict, out_dir: Path):
    """runs: dict tag -> loaded blob. Writes csv + tex."""
    rows = []
    for tag, run in runs.items():
        s = run["summary"]
        e = run["eval"]
        cls_summary = s.get("test_classification", {})
        cons_summary = s.get("test_concepts", {})
        # Prefer tuned values from eval.json
        wf1 = (e.get("classification_tuned", {}).get("weighted_f1")
               or cls_summary.get("weighted_f1") or float("nan"))
        mf1 = (e.get("classification_tuned", {}).get("macro_f1")
               or cls_summary.get("macro_f1") or float("nan"))
        cf1 = (e.get("concept_tuned", {}).get("mean_concept_f1")
               or cons_summary.get("mean_concept_f1") or float("nan"))
        row = dict(
            run=tag,
            weighted_f1=wf1,
            macro_f1=mf1,
            mean_concept_f1=cf1,
            violation_rate=e.get("violation", {}).get("violation_rate", float("nan")),
            probe_subset_acc=e.get("completeness_probe", {}).get("subset_accuracy", float("nan")),
            probe_weighted_f1=e.get("completeness_probe", {}).get("weighted_f1", float("nan")),
            coverage_at_0p05=e.get("conformal", {}).get("marginal_positive_coverage", float("nan")),
            mean_set_size=e.get("conformal", {}).get("mean_set_size", float("nan")),
        )
        rows.append(row)
    csv_path = out_dir / "main_results_table.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    # Simple LaTeX
    tex = ["\\begin{tabular}{l" + "c" * (len(rows[0]) - 1) + "}\\hline"]
    tex.append(" & ".join(rows[0].keys()).replace("_", "\\_") + " \\\\\\hline")
    for r in rows:
        tex.append(" & ".join(
            (v if isinstance(v, str) else f"{v:.3f}") for v in r.values()
        ).replace("_", "\\_") + " \\\\")
    tex.append("\\hline\\end{tabular}")
    (out_dir / "main_results_table.tex").write_text("\n".join(tex))
    print(f"[fig1] table written: {csv_path}")


# ----------------------- 2. per_concept_f1_bar ------------------------------
def fig2_per_concept_f1(primary_run: dict, out_dir: Path,
                          concept_categories: dict | None = None):
    e = primary_run["eval"]
    cons = primary_run["summary"].get("test_concepts", {})
    concepts = primary_run["pred"]["concepts"]
    # Prefer eval.json's tuned per-concept F1 if available; else summary fallback.
    f1s_list = (e.get("concept_tuned", {}).get("per_concept_f1")
                or cons.get("per_concept_f1"))
    if f1s_list is None:
        return
    f1s = np.array(f1s_list)
    order = np.argsort(f1s)
    cats = concept_categories or {}
    cat_color = {"nucleus": "#1f77b4", "chromatin": "#ff7f0e",
                 "granules": "#2ca02c", "cytoplasm": "#d62728"}
    cat_of = {c: cats.get(c, "other") for c in concepts}
    colors = [cat_color.get(cat_of[concepts[i]], "#7f7f7f") for i in order]
    fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(concepts))))
    ax.barh(range(len(order)), f1s[order], color=colors)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([concepts[i] for i in order], fontsize=8)
    ax.axvline(0.9, color="k", lw=0.7, ls="--", label="target 0.90")
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Per-concept F1 (test)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "per_concept_f1_bar.pdf")
    plt.close(fig)
    print(f"[fig2] per_concept_f1_bar.pdf  mean={f1s.mean():.3f}")


# ----------------------- 3. per_concept_f1_comparison -----------------------
def fig3_concept_f1_comparison(runs_by_label: dict[str, dict], out_dir: Path):
    """runs_by_label: e.g. {'BCE (default)': run_no_posw, 'BCE + pos_weight': run_with_posw}.

    Skipped if only one run is available.
    """
    if len(runs_by_label) < 2:
        return
    concepts = next(iter(runs_by_label.values()))["pred"]["concepts"]
    def _per_concept(r):
        return (r["eval"].get("concept_tuned", {}).get("per_concept_f1")
                or r["summary"].get("test_concepts", {}).get("per_concept_f1"))
    f1_by_label = {}
    for lab, r in runs_by_label.items():
        v = _per_concept(r)
        if v is not None:
            f1_by_label[lab] = np.array(v)
    if len(f1_by_label) < 2:
        return
    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(concepts))))
    width = 0.8 / len(f1_by_label)
    y = np.arange(len(concepts))
    for i, (lab, f1) in enumerate(f1_by_label.items()):
        ax.barh(y + i * width, f1, height=width, label=lab)
    ax.set_yticks(y + width * (len(f1_by_label) - 1) / 2)
    ax.set_yticklabels(concepts, fontsize=8)
    ax.axvline(0.9, color="k", lw=0.7, ls="--")
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Per-concept F1 (test)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "per_concept_f1_comparison.pdf")
    plt.close(fig)
    print(f"[fig3] per_concept_f1_comparison.pdf")


# ----------------------- 4. violation_comparison ----------------------------
def fig4_violation_comparison(runs_by_label: dict[str, dict], out_dir: Path):
    """Each run has eval.violation.per_pair_count keyed by 'i-j'."""
    have_pairs = {}
    for lab, r in runs_by_label.items():
        pp = r["eval"].get("violation", {}).get("per_pair_count", {})
        if not pp:
            continue
        have_pairs[lab] = pp
    if len(have_pairs) < 1:
        # Make a placeholder: GR-Neutro prior has no -1 entries, so violation rate is 0 by design.
        fig, ax = plt.subplots(figsize=(5, 2.5))
        ax.text(0.5, 0.5,
                "GR-Neutro prior C has no hard mutex (-1) entries\n"
                "so empirical mutex violations are undefined.\n"
                "Replaced by soft cooccur loss; see eval.cooccurrence.",
                ha="center", va="center", fontsize=10)
        ax.axis("off")
        fig.savefig(out_dir / "violation_comparison.pdf")
        plt.close(fig)
        return
    pair_ids = sorted({k for v in have_pairs.values() for k in v.keys()})
    fig, ax = plt.subplots(figsize=(max(5, 0.4 * len(pair_ids)), 4))
    width = 0.8 / len(have_pairs)
    x = np.arange(len(pair_ids))
    for i, (lab, pp) in enumerate(have_pairs.items()):
        ax.bar(x + i * width, [pp.get(p, 0) for p in pair_ids], width=width, label=lab)
    ax.set_xticks(x + width * (len(have_pairs) - 1) / 2)
    ax.set_xticklabels(pair_ids, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("# violations on test")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "violation_comparison.pdf")
    plt.close(fig)
    print(f"[fig4] violation_comparison.pdf")


# ----------------------- 5. attention_grid ----------------------------------
def attention_to_heatmap(attn_kp: np.ndarray, grid_hw: tuple[int, int]):
    """attn_kp: (P,) attention weights for one (sample, concept) -> 2D heatmap."""
    h, w = grid_hw
    return attn_kp.reshape(h, w)


def fig5_attention_grid(primary_run: dict, out_dir: Path,
                         data_root: Path,
                         filename_lookup: dict,    # idx -> filename
                         classes_to_show: list[str] | None = None):
    pred = primary_run["pred"]
    if "attn" not in pred["test"]:
        print("[fig5] no attn maps in predictions; skipping (baseline run)")
        return
    attn = pred["test"]["attn"].numpy()         # (N, K_concepts, P)
    y = pred["test"]["labels"].numpy().astype(int)
    concepts = pred["concepts"]
    class_names = pred["class_names"]
    # Pick one cell per class (first index where label[k]==1 and only_one_class)
    chosen = []
    chosen_names = []
    for k, cls in enumerate(class_names):
        mask = (y[:, k] == 1) & (y.sum(axis=1) == 1)
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            idxs = np.where(y[:, k] == 1)[0]
        if len(idxs) == 0:
            continue
        chosen.append(int(idxs[0]))
        chosen_names.append(cls)
    if not chosen:
        return
    P = attn.shape[2]
    side = int(round(P ** 0.5))
    grid_hw = (side, side)
    n_show_concepts = min(8, len(concepts))
    # Show: original image + first 8 concepts.
    fig, axes = plt.subplots(len(chosen), 1 + n_show_concepts,
                              figsize=(1.3 * (1 + n_show_concepts), 1.3 * len(chosen)))
    if len(chosen) == 1:
        axes = axes[None, :]
    for i, ci in enumerate(chosen):
        fname = filename_lookup.get(int(pred["test_idx"][ci]), None)
        if fname is None:
            for a in axes[i]:
                a.axis("off")
            axes[i, 0].set_ylabel(chosen_names[i], fontsize=8)
            continue
        img_path = (data_root / "gr_neutro" / chosen_names[i] / fname)
        if not img_path.exists():
            # Fallback: search any folder under data_root for fname
            cand = list(data_root.rglob(fname))
            img_path = cand[0] if cand else None
        if img_path and img_path.exists():
            axes[i, 0].imshow(Image.open(img_path).convert("RGB"))
        axes[i, 0].set_ylabel(chosen_names[i], fontsize=8, rotation=0, ha="right", va="center")
        axes[i, 0].set_xticks([]); axes[i, 0].set_yticks([])
        for j in range(n_show_concepts):
            heat = attention_to_heatmap(attn[ci, j], grid_hw)
            axes[i, 1 + j].imshow(heat, cmap="hot")
            axes[i, 1 + j].set_xticks([]); axes[i, 1 + j].set_yticks([])
            if i == 0:
                axes[i, 1 + j].set_title(concepts[j].replace("_", "\n"), fontsize=6)
    fig.suptitle("Attention maps (rows = cell, columns = concept)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "attention_grid.pdf")
    plt.close(fig)
    print(f"[fig5] attention_grid.pdf  rows={len(chosen)}  concepts={n_show_concepts}")


# ----------------------- 6. conformal_coverage ------------------------------
def fig6_conformal(primary_run: dict, out_dir: Path):
    e = primary_run["eval"].get("conformal", {})
    if not e:
        return
    pcc = e["per_class_coverage"]
    nominal = 1.0 - e["alpha"]
    classes = primary_run["pred"]["class_names"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    # left: empirical vs nominal scatter
    ax = axes[0]
    x = np.arange(len(classes))
    ax.bar(x, pcc, color="#1f77b4")
    ax.axhline(nominal, color="r", ls="--", label=f"nominal {nominal:.2f}")
    ax.set_xticks(x); ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Per-class positive coverage")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=8)
    # right: set size distribution
    ax = axes[1]
    pred_sets = (primary_run["pred"]["test"]["class_logits"].sigmoid().numpy()
                 >= np.array(e["thresholds"])).sum(axis=1)
    ax.hist(pred_sets, bins=range(0, 8), align="left", color="#2ca02c", rwidth=0.8)
    ax.set_xlabel("Prediction set size")
    ax.set_ylabel("# test cells")
    ax.set_title(f"mean={e['mean_set_size']:.2f}")
    fig.suptitle(f"Split-conformal coverage (α={e['alpha']})")
    fig.tight_layout()
    fig.savefig(out_dir / "conformal_coverage.pdf")
    plt.close(fig)
    print(f"[fig6] conformal_coverage.pdf")


# ----------------------- 7. cooccur_vs_prior --------------------------------
def fig7_cooccur(primary_run: dict, out_dir: Path):
    e = primary_run["eval"].get("cooccurrence", {})
    if not e:
        return
    emp = np.array(e["empirical_cooccurrence"])
    prior = np.array(e["prior_C"])
    concepts = primary_run["pred"]["concepts"]
    short = ["lob","cnt","nc","cnd","clm","grd","crs","tex","bso","inc","vac"][:len(concepts)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, M, title in [(axes[0], prior, "Prior C"),
                         (axes[1], emp, "Empirical cooccurrence (test)")]:
        im = ax.imshow(M, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(len(short))); ax.set_yticks(range(len(short)))
        ax.set_xticklabels(short, fontsize=7, rotation=45)
        ax.set_yticklabels(short, fontsize=7)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"Pearson(off-diag) = {e['off_diag_pearson_corr']:.3f}")
    fig.tight_layout()
    fig.savefig(out_dir / "cooccur_vs_prior.pdf")
    plt.close(fig)
    print(f"[fig7] cooccur_vs_prior.pdf")


# ----------------------- 8. confusion_matrix --------------------------------
def fig8_confusion(primary_run: dict, out_dir: Path):
    pred = primary_run["pred"]
    cls_p = pred["test"]["class_logits"].sigmoid().numpy()
    y = pred["test"]["labels"].numpy().astype(int)
    classes = pred["class_names"]
    K = y.shape[1]
    pred_h = (cls_p >= 0.5).astype(int)
    # Multi-label confusion: rows = true class k, cols = predicted class j; entry = # cells where y_k=1 AND pred_j=1
    cm = np.zeros((K, K), dtype=float)
    for i in range(len(y)):
        true_k = np.where(y[i] == 1)[0]
        pred_k = np.where(pred_h[i] == 1)[0]
        for tk in true_k:
            for pk in pred_k:
                cm[tk, pk] += 1
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(K):
        for j in range(K):
            ax.text(j, i, f"{cm_norm[i,j]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if cm_norm[i,j] > 0.5 else "black")
    ax.set_xticks(range(K)); ax.set_yticks(range(K))
    ax.set_xticklabels(classes, fontsize=7, rotation=45, ha="right")
    ax.set_yticklabels(classes, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Row-normalized multi-label confusion (test)")
    plt.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.pdf")
    plt.close(fig)
    print(f"[fig8] confusion_matrix.pdf")


# ----------------------- 9. training_curves ---------------------------------
def fig9_training_curves(primary_run: dict, out_dir: Path):
    log = primary_run["log"]
    if len(log) <= 1:
        return
    header, *rows = log
    rows = [[float(x) if x not in ("", "nan") else float("nan") for x in r] for r in rows]
    a = np.array(rows)
    cols = {h: i for i, h in enumerate(header)}
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    ep = a[:, cols["epoch"]]
    axes[0].plot(ep, a[:, cols["train_total"]], label="train total")
    axes[0].plot(ep, a[:, cols["train_bce_cls"]], label="bce_cls")
    if "train_bce_con" in cols:
        axes[0].plot(ep, a[:, cols["train_bce_con"]], label="bce_con")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss"); axes[0].legend(fontsize=8)
    axes[1].plot(ep, a[:, cols["train_class_f1"]], label="train")
    axes[1].plot(ep, a[:, cols["val_class_f1"]], label="val")
    axes[1].axhline(0.9, color="k", ls="--", lw=0.5)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("classification macro F1"); axes[1].legend(fontsize=8)
    if "val_concept_f1" in cols:
        axes[2].plot(ep, a[:, cols["train_concept_f1"]], label="train")
        axes[2].plot(ep, a[:, cols["val_concept_f1"]], label="val")
        axes[2].axhline(0.9, color="k", ls="--", lw=0.5)
        axes[2].set_xlabel("epoch"); axes[2].set_ylabel("concept F1"); axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.pdf")
    plt.close(fig)
    print(f"[fig9] training_curves.pdf")


# ----------------------- 10. lambda_pareto ----------------------------------
def fig10_lambda_pareto(sweep_runs: list[dict], out_dir: Path):
    if not sweep_runs:
        return
    lam = []; vio = []; f1 = []; cf1 = []
    for r in sweep_runs:
        lam.append(r["summary"]["args"]["lambda_constraint"])
        vio.append(r["eval"].get("violation", {}).get("violation_rate", float("nan")))
        f1.append(r["eval"].get("classification_tuned", {}).get("weighted_f1")
                  or r["summary"].get("test_classification", {}).get("weighted_f1") or float("nan"))
        cf1.append(r["eval"].get("concept_tuned", {}).get("mean_concept_f1")
                   or r["summary"].get("test_concepts", {}).get("mean_concept_f1") or float("nan"))
    order = np.argsort(lam)
    lam = np.array(lam)[order]; vio = np.array(vio)[order]
    f1 = np.array(f1)[order]; cf1 = np.array(cf1)[order]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].plot(lam, f1, "o-", label="classification weighted F1")
    axes[0].plot(lam, cf1, "s-", label="mean concept F1")
    axes[0].axhline(0.9, color="k", ls="--", lw=0.5)
    axes[0].set_xscale("symlog", linthresh=0.005)
    axes[0].set_xlabel("λ_constraint"); axes[0].set_ylabel("F1")
    axes[0].legend(fontsize=8)
    axes[1].plot(f1, vio, "o-")
    for i, l in enumerate(lam):
        axes[1].annotate(f"λ={l:g}", (f1[i], vio[i]), fontsize=7)
    axes[1].set_xlabel("classification weighted F1")
    axes[1].set_ylabel("soft violation rate")
    fig.tight_layout()
    fig.savefig(out_dir / "lambda_pareto.pdf")
    plt.close(fig)
    print(f"[fig10] lambda_pareto.pdf  λ values: {lam.tolist()}")


# ----------------------- E1. misclass_examples ------------------------------
def figE1_misclass(primary_run: dict, out_dir: Path,
                    data_root: Path, filename_lookup: dict, n: int = 12):
    pred = primary_run["pred"]
    cls_p = pred["test"]["class_logits"].sigmoid().numpy()
    y = pred["test"]["labels"].numpy().astype(int)
    pred_h = (cls_p >= 0.5).astype(int)
    classes = pred["class_names"]
    concepts = pred["concepts"]
    # misclass: any class where prediction differs from truth
    wrong = np.where((pred_h != y).any(axis=1))[0]
    if len(wrong) == 0:
        return
    rng = np.random.default_rng(0)
    pick = rng.choice(wrong, size=min(n, len(wrong)), replace=False)
    has_attn = "attn" in pred["test"]
    n_concepts_show = min(6, len(concepts))
    cols = 1 + n_concepts_show + 1
    fig, axes = plt.subplots(len(pick), cols,
                              figsize=(1.3 * cols, 1.3 * len(pick)))
    if len(pick) == 1:
        axes = axes[None, :]
    for r, ci in enumerate(pick):
        fname = filename_lookup.get(int(pred["test_idx"][ci]), None)
        img_path = None
        if fname:
            cand = list(data_root.rglob(fname))
            img_path = cand[0] if cand else None
        if img_path and img_path.exists():
            axes[r, 0].imshow(Image.open(img_path).convert("RGB"))
        axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        true_lab = ",".join(classes[k] for k in np.where(y[ci]==1)[0])
        pred_lab = ",".join(classes[k] for k in np.where(pred_h[ci]==1)[0]) or "(none)"
        axes[r, 0].set_ylabel(f"T:{true_lab}\nP:{pred_lab}", fontsize=6, rotation=0,
                                ha="right", va="center")
        if has_attn:
            attn = pred["test"]["attn"].numpy()
            P = attn.shape[2]; side = int(round(P ** 0.5))
            for j in range(n_concepts_show):
                axes[r, 1 + j].imshow(attn[ci, j].reshape(side, side), cmap="hot")
                axes[r, 1 + j].set_xticks([]); axes[r, 1 + j].set_yticks([])
                if r == 0:
                    axes[r, 1 + j].set_title(concepts[j][:8], fontsize=6)
        # Last col: bar of concept probs
        con_p = pred["test"]["concept_logits"].sigmoid().numpy()
        axes[r, -1].barh(range(len(concepts)), con_p[ci], color="steelblue")
        axes[r, -1].set_xlim(0, 1)
        axes[r, -1].axvline(0.5, color="k", lw=0.5)
        axes[r, -1].set_yticks(range(len(concepts)))
        axes[r, -1].set_yticklabels(concepts, fontsize=5)
        axes[r, -1].invert_yaxis()
    fig.suptitle("Misclassified test cells with attention + concept probs", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "misclass_examples.pdf")
    plt.close(fig)
    print(f"[figE1] misclass_examples.pdf  n={len(pick)}")


# ----------------------- E2. concept_exemplars ------------------------------
def figE2_concept_exemplars(primary_run: dict, out_dir: Path,
                              data_root: Path, filename_lookup: dict, top_k: int = 3):
    pred = primary_run["pred"]
    con_p = pred["test"]["concept_logits"].sigmoid().numpy()
    test_idx = pred["test_idx"]
    concepts = pred["concepts"]
    classes = pred["class_names"]
    y = pred["test"]["labels"].numpy().astype(int)
    fig, axes = plt.subplots(len(concepts), top_k,
                              figsize=(1.4 * top_k, 1.4 * len(concepts)))
    if top_k == 1:
        axes = axes[:, None]
    for c, name in enumerate(concepts):
        order = np.argsort(con_p[:, c])[::-1][:top_k]
        for j, ci in enumerate(order):
            fname = filename_lookup.get(int(test_idx[ci]), None)
            cand = list(data_root.rglob(fname)) if fname else []
            img = Image.open(cand[0]).convert("RGB") if cand else None
            ax = axes[c, j]
            if img is not None:
                ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            true_lab = ",".join(classes[k] for k in np.where(y[ci]==1)[0])[:12]
            ax.set_title(f"p={con_p[ci, c]:.2f}\n{true_lab}", fontsize=5)
        axes[c, 0].set_ylabel(name, fontsize=6, rotation=0, ha="right", va="center")
    fig.suptitle("Top-k test cells per concept", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "concept_exemplars.pdf")
    plt.close(fig)
    print(f"[figE2] concept_exemplars.pdf  top_k={top_k}")


# ----------------------- main -----------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default=str(HERE / "outputs"))
    ap.add_argument("--fig_dir", default=str(HERE / "figures"))
    ap.add_argument("--data_root", default=str(WORKDIR / "data" / "gr_neutro_extended"))
    ap.add_argument("--csv", default=str(WORKDIR / "data" / "gr_neutro_extended" / "annotations.csv"))
    ap.add_argument("--primary_tag", required=True)
    ap.add_argument("--baseline_tag", default=None)
    ap.add_argument("--unconstrained_tag", default=None)
    ap.add_argument("--lambda_sweep_tags", default="",
                    help="Comma-separated list of tags constituting the lambda sweep.")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)

    # Build filename lookup: test_idx (global) -> filename
    rows_by_idx = {}
    with open(args.csv) as f:
        r = csv.reader(f)
        next(r)
        for i, row in enumerate(r):
            rows_by_idx[i] = row[0]

    primary = load_run(args.primary_tag, out_root)
    main_runs = {args.primary_tag: primary}
    if args.baseline_tag:
        main_runs[args.baseline_tag] = load_run(args.baseline_tag, out_root)
    if args.unconstrained_tag:
        main_runs[args.unconstrained_tag] = load_run(args.unconstrained_tag, out_root)

    fig1_main_table(main_runs, fig_dir)

    concept_categories = {
        "nuclear_lobulation_degree": "nucleus", "nuclear_contour_irregularity": "nucleus",
        "nucleus_to_cytoplasm_ratio": "nucleus",
        "chromatin_condensation_level": "chromatin", "chromatin_clumping_pattern": "chromatin",
        "cytoplasmic_granule_density": "granules", "granule_coarseness": "granules",
        "cytoplasmic_texture_uniformity": "cytoplasm", "cytoplasm_basophilia_level": "cytoplasm",
        "cytoplasmic_inclusion_visibility": "cytoplasm", "cytoplasmic_vacuolization_degree": "cytoplasm",
    }
    fig2_per_concept_f1(primary, fig_dir, concept_categories=concept_categories)

    cmp_runs = {}
    if args.unconstrained_tag:
        cmp_runs["constrained"] = primary
        cmp_runs["unconstrained"] = main_runs[args.unconstrained_tag]
        fig3_concept_f1_comparison(cmp_runs, fig_dir)
        fig4_violation_comparison(cmp_runs, fig_dir)
    else:
        fig4_violation_comparison({args.primary_tag: primary}, fig_dir)

    fig5_attention_grid(primary, fig_dir, data_root, rows_by_idx)
    fig6_conformal(primary, fig_dir)
    fig7_cooccur(primary, fig_dir)
    fig8_confusion(primary, fig_dir)
    fig9_training_curves(primary, fig_dir)

    if args.lambda_sweep_tags:
        sweep = [load_run(t.strip(), out_root) for t in args.lambda_sweep_tags.split(",") if t.strip()]
        fig10_lambda_pareto(sweep, fig_dir)

    figE1_misclass(primary, fig_dir, data_root, rows_by_idx)
    figE2_concept_exemplars(primary, fig_dir, data_root, rows_by_idx)

    # Tar everything
    import tarfile
    tar_path = fig_dir.parent / "figures.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        for p in fig_dir.iterdir():
            tar.add(p, arcname=p.name)
    print(f"[tar] {tar_path}")


if __name__ == "__main__":
    main()
