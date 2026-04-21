"""Evaluate trained models: per-concept metrics, violation rate, completeness probe,
attention maps, conformal prediction, learned co-occurrence matrix.

Usage:
  python evaluate.py --run dinobloom_s_aml_matek_joint_const
  python evaluate.py --all
"""

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

WORKDIR = Path("/gpfs/workdir/mouaddenn")
THESIS = WORKDIR / "thesis" / "aml_matek"
OUTPUTS = THESIS / "outputs"


def load_concept_cfg():
    return json.loads((THESIS / "concept_config.json").read_text())


# ---------------------------------------------------------------------------
# 7a. Per-concept metrics
# ---------------------------------------------------------------------------
def concept_metrics(pred_sigmoid: np.ndarray, target_soft: np.ndarray, concepts: list[str], out_dir: Path):
    """pred_sigmoid: (N, K) sigmoid outputs; target_soft: (N, K) soft targets in [0,1]."""
    y_true = (target_soft >= 0.5).astype(int)
    y_pred = (pred_sigmoid >= 0.5).astype(int)

    rows = []
    for k, name in enumerate(concepts):
        t = y_true[:, k]
        p = y_pred[:, k]
        if t.sum() == 0 or t.sum() == len(t):
            f1 = prec = rec = auroc = float("nan")
        else:
            f1 = f1_score(t, p, zero_division=0)
            prec = precision_score(t, p, zero_division=0)
            rec = recall_score(t, p, zero_division=0)
            try:
                auroc = roc_auc_score(t, pred_sigmoid[:, k])
            except ValueError:
                auroc = float("nan")
        rows.append(dict(concept=name, f1=f1, precision=prec, recall=rec, auroc=auroc,
                         support=int(t.sum())))

    csv_path = out_dir / "concept_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["concept", "f1", "precision", "recall", "auroc", "support"])
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})

    macro_f1 = np.nanmean([r["f1"] for r in rows])
    mean_auroc = np.nanmean([r["auroc"] for r in rows])

    # Bar chart of F1 per concept.
    f1s = [r["f1"] if not math.isnan(r["f1"]) else 0.0 for r in rows]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(concepts)), f1s)
    ax.set_xticks(range(len(concepts)))
    ax.set_xticklabels(concepts, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("F1 @ 0.5")
    ax.set_ylim(0, 1)
    ax.set_title(f"Per-concept F1 (macro F1 = {macro_f1:.3f})")
    fig.tight_layout()
    fig.savefig(out_dir / "concept_f1_bar.png", dpi=120)
    plt.close(fig)

    return dict(macro_f1=macro_f1, mean_auroc=mean_auroc)


# ---------------------------------------------------------------------------
# 7b. Violation rate
# ---------------------------------------------------------------------------
def violation_rate(pred_sigmoid: np.ndarray, concepts: list[str], cfg: dict, out_dir: Path):
    idx = {c: i for i, c in enumerate(concepts)}
    pairs = [(idx[p["concepts"][0]], idx[p["concepts"][1]])
             for p in cfg["concept_constraint_matrix"]["mutually_exclusive_pairs"]]
    hard = (pred_sigmoid >= 0.5)
    n_samples = hard.shape[0]
    per_pair = []
    total = 0
    for a, b in pairs:
        v = int(((hard[:, a] & hard[:, b])).sum())
        per_pair.append((concepts[a], concepts[b], v, v / n_samples))
        total += v

    rate = total / (n_samples * max(1, len(pairs)))

    with open(out_dir / "violation_rate.txt", "w") as f:
        f.write(f"n_samples={n_samples}\n")
        f.write(f"n_exclusive_pairs={len(pairs)}\n")
        f.write(f"total_violations={total}\n")
        f.write(f"violation_rate={rate:.6f}\n\n")
        f.write("per_pair (concept_a, concept_b, count, rate_per_sample):\n")
        for a, b, v, r in per_pair:
            f.write(f"  {a:28s} ^ {b:28s}  count={v}  rate={r:.4f}\n")
    return rate


# ---------------------------------------------------------------------------
# 7c. Completeness probe
# ---------------------------------------------------------------------------
def completeness_probe(pred_cal, y_cal, pred_test, y_test, direct_acc: float, out_dir: Path):
    clf = LogisticRegression(max_iter=2000, n_jobs=-1)
    clf.fit(pred_cal, y_cal)
    probe_acc = clf.score(pred_test, y_test)
    with open(out_dir / "completeness_probe.txt", "w") as f:
        f.write(f"direct_classifier_accuracy={direct_acc:.4f}\n")
        f.write(f"probe_accuracy_from_concepts={probe_acc:.4f}\n")
        f.write(f"gap={direct_acc - probe_acc:+.4f}\n")
    return probe_acc


# ---------------------------------------------------------------------------
# 7d. Cross-attention maps
# ---------------------------------------------------------------------------
def save_attention_maps(run_dir: Path, concepts: list[str]):
    path = run_dir / "attention_weights.pt"
    if not path.exists():
        return
    blob = torch.load(path, map_location="cpu", weights_only=False)
    attn = blob["attn"].numpy()              # (n, K, P)
    files = blob["filenames"]
    labels = blob["labels"].numpy()
    H, W = blob["grid_hw"]
    out_sub = run_dir / "attention_maps"
    out_sub.mkdir(parents=True, exist_ok=True)

    # Save a grid per image: original + K attention heatmaps.
    for i, fp in enumerate(files):
        try:
            img = Image.open(fp).convert("RGB").resize((224, 224))
        except Exception:
            continue
        K = attn.shape[1]
        n_cols = 6
        n_rows = math.ceil((K + 1) / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.2, n_rows * 2.4))
        axes = np.array(axes).reshape(-1)
        axes[0].imshow(img)
        axes[0].set_title(f"img (label={labels[i]})", fontsize=8)
        axes[0].axis("off")
        for k in range(K):
            a = attn[i, k].reshape(H, W)
            a_up = np.array(Image.fromarray((a * 255 / max(a.max(), 1e-9)).astype(np.uint8))
                            .resize((224, 224), Image.BILINEAR))
            axes[k + 1].imshow(img)
            axes[k + 1].imshow(a_up, cmap="jet", alpha=0.45)
            axes[k + 1].set_title(concepts[k][:14], fontsize=7)
            axes[k + 1].axis("off")
        for j in range(K + 1, len(axes)):
            axes[j].axis("off")
        fig.tight_layout()
        fname = Path(fp).stem
        fig.savefig(out_sub / f"{i:02d}_{fname}.png", dpi=110)
        plt.close(fig)


# ---------------------------------------------------------------------------
# 7e. Conformal prediction per concept
# ---------------------------------------------------------------------------
def conformal_per_concept(pred_cal: np.ndarray, tgt_cal: np.ndarray,
                          pred_test: np.ndarray, tgt_test: np.ndarray,
                          concepts: list[str], out_dir: Path, alphas=(0.01, 0.05, 0.10)):
    """For each concept, compute a per-concept prediction set: detected / absent / uncertain.

    Non-conformity score at (sample, concept):
      s = 1 - p_hat  if concept is present (target>=0.5)
      s = p_hat      if concept is absent
    Then q_hat at quantile ceil((1-alpha)(n+1))/n of calibration scores.
    Decision rule at test time:
      p_hat > 1 - q_hat -> detected
      p_hat < q_hat     -> absent
      otherwise         -> uncertain
    """
    K = pred_cal.shape[1]
    tgt_cal_bin = (tgt_cal >= 0.5).astype(int)
    tgt_test_bin = (tgt_test >= 0.5).astype(int)

    rows = []
    for alpha in alphas:
        per_concept_cov = []
        per_concept_unc = []
        for k in range(K):
            pc, tc = pred_cal[:, k], tgt_cal_bin[:, k]
            s_cal = np.where(tc == 1, 1 - pc, pc)
            n = len(s_cal)
            q = np.quantile(s_cal, min(1.0, math.ceil((1 - alpha) * (n + 1)) / n), method="higher")
            pt, tt = pred_test[:, k], tgt_test_bin[:, k]
            detected = pt > 1 - q
            absent = pt < q
            uncertain = ~(detected | absent)
            # Coverage: prediction is "covered" if either (1) target=1 and detected,
            # (2) target=0 and absent, or (3) uncertain (the set contains both options).
            covered = ((tt == 1) & detected) | ((tt == 0) & absent) | uncertain
            per_concept_cov.append(float(covered.mean()))
            per_concept_unc.append(float(uncertain.mean()))
        rows.append(dict(
            alpha=alpha,
            mean_coverage=float(np.mean(per_concept_cov)),
            min_coverage=float(np.min(per_concept_cov)),
            mean_uncertain_fraction=float(np.mean(per_concept_unc)),
        ))
        # Per-concept detail
        detail_path = out_dir / f"conformal_detail_alpha{alpha}.csv"
        with open(detail_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["concept", "coverage", "uncertain_frac"])
            for k, name in enumerate(concepts):
                w.writerow([name, f"{per_concept_cov[k]:.4f}", f"{per_concept_unc[k]:.4f}"])

    with open(out_dir / "conformal_coverage.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["alpha", "mean_coverage", "min_coverage", "mean_uncertain_fraction"])
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    return rows


# ---------------------------------------------------------------------------
# 7f. Learned co-occurrence
# ---------------------------------------------------------------------------
def learned_cooccurrence(pred_sigmoid: np.ndarray, concepts: list[str],
                          prior_C: np.ndarray, out_dir: Path):
    """Correlation of predicted sigmoid probabilities, K x K."""
    # Pearson correlation of columns
    p = pred_sigmoid - pred_sigmoid.mean(0, keepdims=True)
    denom = np.sqrt((p ** 2).sum(0, keepdims=True))
    denom = np.where(denom < 1e-9, 1e-9, denom)
    corr = (p.T @ p) / (denom.T @ denom)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(corr, ax=axes[0], xticklabels=concepts, yticklabels=concepts,
                cmap="coolwarm", vmin=-1, vmax=1, cbar=True)
    axes[0].set_title("Learned co-occurrence (Pearson corr)")
    sns.heatmap(prior_C, ax=axes[1], xticklabels=concepts, yticklabels=concepts,
                cmap="coolwarm", vmin=-1, vmax=1, cbar=True)
    axes[1].set_title("Prior constraint matrix C")
    for ax in axes:
        ax.tick_params(axis="x", rotation=60, labelsize=7)
        ax.tick_params(axis="y", rotation=0, labelsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "constraint_prior_vs_learned.png", dpi=110)
    plt.close(fig)

    # Also save standalone learned-only heatmap.
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, ax=ax, xticklabels=concepts, yticklabels=concepts,
                cmap="coolwarm", vmin=-1, vmax=1, cbar=True)
    ax.set_title("Learned co-occurrence matrix")
    ax.tick_params(axis="x", rotation=60, labelsize=7)
    ax.tick_params(axis="y", rotation=0, labelsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "cooccurrence_heatmap.png", dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
def evaluate_one(run_dir: Path):
    print(f"\n=== evaluate {run_dir.name} ===")
    pred_path = run_dir / "predictions.pt"
    if not pred_path.exists():
        print(f"  skip (no predictions.pt)")
        return
    blob = torch.load(pred_path, map_location="cpu", weights_only=False)
    if blob.get("is_baseline"):
        print(f"  skip (baseline — concept metrics N/A)")
        return
    concepts = blob["concepts"]
    num_classes = blob["num_classes"]

    from models import build_prior_C
    cfg = load_concept_cfg()
    prior_C = build_prior_C(concepts, cfg["concept_constraint_matrix"]).numpy()

    test = blob["test"]; cal = blob["cal"]
    test_pred_sig = test["concept_logits"].sigmoid().numpy()
    cal_pred_sig = cal["concept_logits"].sigmoid().numpy()
    test_tgt = test["concept_targets"].numpy()
    cal_tgt = cal["concept_targets"].numpy()
    y_test = test["labels"].numpy()
    y_cal = cal["labels"].numpy()

    # 7a
    a = concept_metrics(test_pred_sig, test_tgt, concepts, run_dir)
    # 7b
    vr = violation_rate(test_pred_sig, concepts, cfg, run_dir)
    # direct classifier accuracy (for completeness comparison)
    direct_acc = float((test["class_logits"].argmax(-1).numpy() == y_test).mean())
    # 7c
    probe_acc = completeness_probe(cal_pred_sig, y_cal, test_pred_sig, y_test, direct_acc, run_dir)
    # 7d
    save_attention_maps(run_dir, concepts)
    # 7e
    conformal_rows = conformal_per_concept(cal_pred_sig, cal_tgt, test_pred_sig, test_tgt, concepts, run_dir)
    # 7f
    learned_cooccurrence(test_pred_sig, concepts, prior_C, run_dir)

    # Aggregate summary
    summary = dict(
        run=run_dir.name,
        macro_f1=a["macro_f1"],
        mean_auroc=a["mean_auroc"],
        violation_rate=vr,
        direct_class_acc=direct_acc,
        probe_from_concepts_acc=probe_acc,
        conformal=conformal_rows,
    )
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  macro_f1={a['macro_f1']:.3f}  mean_auroc={a['mean_auroc']:.3f}  "
          f"violation_rate={vr:.4f}  direct_acc={direct_acc:.3f}  probe_acc={probe_acc:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="specific output dir name under outputs/")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).parent))

    if args.all:
        runs = sorted([d for d in OUTPUTS.iterdir() if d.is_dir()])
    elif args.run:
        runs = [OUTPUTS / args.run]
    else:
        ap.error("pass --run or --all")

    for r in runs:
        try:
            evaluate_one(r)
        except Exception as e:
            print(f"  FAILED {r.name}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
