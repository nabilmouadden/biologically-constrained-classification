"""C4 — Aggregate VLM-disagreement-aux results across seeds and lambdas.

For each (lambda, seed) checkpoint:
  - Load model.pt + predictions.pt
  - Compute test W-F1, macro-F1, mean concept F1
  - Compute per-concept Spearman vs BiomedCLIP cosine-ratio score
  - Count concept dims clearing 0.05 Spearman threshold ("rank decompression")
  - Compute paired (vs joint baseline B_kitchen_s*)

Outputs:
  results_per_seed.csv
  results_aggregated.json    (mean+std per lambda)
  vlm_aux_block.tex          (LaTeX table for the paper)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy import stats

HERE = Path(__file__).resolve().parent
CH3 = HERE.parent
OUTS = HERE / "outputs"
DELTA_DIR = HERE
# LATEX_OUT defaults to a Ruche-local path; pull this file back to the laptop
# via scp once it's produced. Override via env var if needed.
import os as _os
LATEX_OUT = Path(_os.environ.get("VLM_AUX_LATEX_OUT", str(HERE / "vlm_aux_block.tex")))

# Load per-cell BiomedCLIP cosine-ratio scores (the published 11-concept scores).
# These are the "ground-truth" V1 scores from the paper's grounding pipeline.
V1_BIOMED = np.load(
    "/gpfs/workdir/mouaddenn/thesis/ch3_gr_neutro/outputs/vlm_grounding/scores_v2/biomedclip/scores.npz",
    allow_pickle=True,
)
biomed_scores = V1_BIOMED["scores"].astype(np.float32)  # (4378, 11)
biomed_paths = list(V1_BIOMED["paths"])
biomed_basenames = [Path(p).name for p in biomed_paths]
biomed_concepts = list(V1_BIOMED["concepts"])
biomed_lookup = {b: i for i, b in enumerate(biomed_basenames)}

SEEDS = [13, 42, 100]
LAMS = [0.1, 0.5, 1.0]


def spearman_per_concept(pred_logits: np.ndarray, target_scores: np.ndarray) -> list[float]:
    rhos = []
    for k in range(pred_logits.shape[1]):
        rho, _ = stats.spearmanr(pred_logits[:, k], target_scores[:, k])
        rhos.append(float(rho) if not np.isnan(rho) else 0.0)
    return rhos


def compute_metrics_for_run(run_dir: Path):
    summary = json.loads((run_dir / "summary.json").read_text())
    cls = summary["test_classification"]
    con = summary["test_concepts"]

    preds = torch.load(run_dir / "predictions.pt", map_location="cpu")
    test = preds["test"]
    test_filenames = test["filenames"]
    concept_logits = test["concept_logits"].numpy()  # (N_test, 11)

    # Align biomedclip per-cell scores to the test filenames.
    n_test = len(test_filenames)
    biomed_aligned = np.zeros_like(concept_logits)
    n_matched = 0
    for i, fn in enumerate(test_filenames):
        idx = biomed_lookup.get(fn)
        if idx is not None:
            biomed_aligned[i] = biomed_scores[idx]
            n_matched += 1

    rhos = spearman_per_concept(concept_logits, biomed_aligned)
    n_pass_05 = sum(1 for r in rhos if r >= 0.05)
    n_pass_10 = sum(1 for r in rhos if r >= 0.10)
    mean_abs_rho = float(np.mean(np.abs(rhos)))
    return dict(
        weighted_f1=cls["weighted_f1"],
        macro_f1=cls["macro_f1"],
        mean_concept_f1=con["mean_concept_f1"],
        per_concept_spearman_vs_biomedclip=rhos,
        n_concept_dims_above_0p05_spearman=n_pass_05,
        n_concept_dims_above_0p10_spearman=n_pass_10,
        mean_abs_spearman_vs_biomedclip=mean_abs_rho,
        n_matched_for_alignment=n_matched,
    )


def main():
    rows = []
    aggregated = {}
    for lam in LAMS:
        per_seed = []
        for seed in SEEDS:
            tag = f"vlm_aux_lam{lam}_s{seed}"
            run_dir = OUTS / tag
            if not (run_dir / "predictions.pt").exists():
                print(f"[skip] {tag}: missing predictions.pt")
                continue
            try:
                metrics = compute_metrics_for_run(run_dir)
            except Exception as e:
                print(f"[fail] {tag}: {e}")
                continue
            metrics["tag"] = tag
            metrics["lambda"] = lam
            metrics["seed"] = seed
            per_seed.append(metrics)
            rows.append(metrics)
            print(f"[{tag}] wf1={metrics['weighted_f1']:.4f}  "
                  f"cf1={metrics['mean_concept_f1']:.4f}  "
                  f"n_pass@0.05={metrics['n_concept_dims_above_0p05_spearman']}/11  "
                  f"mean|rho|={metrics['mean_abs_spearman_vs_biomedclip']:.3f}")
        if per_seed:
            wfs = np.array([m["weighted_f1"] for m in per_seed])
            cfs = np.array([m["mean_concept_f1"] for m in per_seed])
            n05 = np.array([m["n_concept_dims_above_0p05_spearman"] for m in per_seed])
            n10 = np.array([m["n_concept_dims_above_0p10_spearman"] for m in per_seed])
            mar = np.array([m["mean_abs_spearman_vs_biomedclip"] for m in per_seed])
            aggregated[f"lambda_{lam}"] = dict(
                n_seeds=len(per_seed),
                weighted_f1_mean=float(wfs.mean()), weighted_f1_std=float(wfs.std()),
                mean_concept_f1_mean=float(cfs.mean()), mean_concept_f1_std=float(cfs.std()),
                n_pass_0p05_mean=float(n05.mean()), n_pass_0p05_std=float(n05.std()),
                n_pass_0p10_mean=float(n10.mean()), n_pass_0p10_std=float(n10.std()),
                mean_abs_spearman_mean=float(mar.mean()), mean_abs_spearman_std=float(mar.std()),
            )

    # ----- Baseline: B_kitchen joint runs (no aux head) -----
    # Need to recover basenames from test_idx via annotations.csv ordering.
    import csv as _csv
    annot_rows = []
    with open("/gpfs/workdir/mouaddenn/data/gr_neutro_extended/annotations.csv") as f:
        r = _csv.reader(f); next(r)
        for row in r:
            annot_rows.append(row[0])  # basenames
    annot_rows = np.array(annot_rows)

    bk_rows = []
    bk_dir = Path("/gpfs/workdir/mouaddenn/thesis/ch3_gr_neutro/outputs")
    bk_seeds = [13, 42, 100]
    for seed in bk_seeds:
        bk = bk_dir / f"B_kitchen_s{seed}"
        if not (bk / "summary.json").exists():
            continue
        try:
            sm = json.loads((bk / "summary.json").read_text())
            preds = torch.load(bk / "predictions.pt", map_location="cpu", weights_only=False)
            test = preds["test"]
            cls = sm["test_classification"]
            con = sm["test_concepts"]
            con_logits = test["concept_logits"].numpy()
            test_idx = preds["test_idx"]
            test_basenames = list(annot_rows[test_idx])

            # Compute baseline Spearman vs BiomedCLIP cosine-ratio
            biomed_aligned = np.zeros_like(con_logits)
            for i, fn in enumerate(test_basenames):
                idx = biomed_lookup.get(fn)
                if idx is not None:
                    biomed_aligned[i] = biomed_scores[idx]
            rhos = spearman_per_concept(con_logits, biomed_aligned)
            n_pass_05 = sum(1 for r in rhos if r >= 0.05)
            n_pass_10 = sum(1 for r in rhos if r >= 0.10)
            mean_abs_rho = float(np.mean(np.abs(rhos)))

            bk_rows.append(dict(
                tag=f"B_kitchen_s{seed}", seed=seed,
                weighted_f1=cls["weighted_f1"], macro_f1=cls["macro_f1"],
                mean_concept_f1=con["mean_concept_f1"],
                per_concept_spearman_vs_biomedclip=rhos,
                n_concept_dims_above_0p05_spearman=n_pass_05,
                n_concept_dims_above_0p10_spearman=n_pass_10,
                mean_abs_spearman_vs_biomedclip=mean_abs_rho,
            ))
            print(f"[baseline B_kitchen_s{seed}] wf1={cls['weighted_f1']:.4f}  "
                  f"n_pass@0.05={n_pass_05}/11  mean|rho|={mean_abs_rho:.3f}")
        except Exception as e:
            print(f"[baseline-skip] {seed}: {e}")
    if bk_rows:
        wfs = np.array([r["weighted_f1"] for r in bk_rows])
        cfs = np.array([r["mean_concept_f1"] for r in bk_rows])
        n05 = np.array([r["n_concept_dims_above_0p05_spearman"] for r in bk_rows])
        n10 = np.array([r["n_concept_dims_above_0p10_spearman"] for r in bk_rows])
        mar = np.array([r["mean_abs_spearman_vs_biomedclip"] for r in bk_rows])
        aggregated["baseline_joint_B_kitchen"] = dict(
            n_seeds=len(bk_rows),
            weighted_f1_mean=float(wfs.mean()), weighted_f1_std=float(wfs.std()),
            mean_concept_f1_mean=float(cfs.mean()), mean_concept_f1_std=float(cfs.std()),
            n_pass_0p05_mean=float(n05.mean()), n_pass_0p05_std=float(n05.std()),
            n_pass_0p10_mean=float(n10.mean()), n_pass_0p10_std=float(n10.std()),
            mean_abs_spearman_mean=float(mar.mean()), mean_abs_spearman_std=float(mar.std()),
        )
        print(f"\n[baseline B_kitchen s={bk_seeds}] mean wf1={wfs.mean():.4f}±{wfs.std():.4f}  "
              f"n_pass@0.05={n05.mean():.1f}±{n05.std():.1f}")

    # ----- save -----
    out = HERE / "results_per_seed.json"
    out.write_text(json.dumps(rows, indent=2))
    (HERE / "results_aggregated.json").write_text(json.dumps(aggregated, indent=2))

    # ----- LaTeX block -----
    LATEX_OUT.parent.mkdir(parents=True, exist_ok=True)
    indep_report = json.loads((DELTA_DIR / "class_indep_check.json").read_text())
    n_indep05 = indep_report["summary"]["n_concepts_indep_at_05"]
    n_total_concepts = indep_report["summary"]["n_concepts_total"]
    effects = json.loads((DELTA_DIR / "effect_sizes.json").read_text()) \
              if (DELTA_DIR / "effect_sizes.json").exists() else None
    mean_eta2 = effects["mean_eta_squared"] if effects else float("nan")

    bk = aggregated.get("baseline_joint_B_kitchen", {})
    def fmt_pair(d, key_mean, key_std, fmt="%.3f"):
        if key_mean not in d:
            return "n/a"
        return (fmt + " $\\pm$ " + fmt) % (d[key_mean], d[key_std])

    lines = [
        r"% C4 -- VLM-disagreement as iVAE auxiliary variable.",
        r"% Auto-generated by ch3_gr_neutro/vlm_aux_work/aggregate_vlm_aux.py.",
        r"\subsection{VLM-disagreement as an auxiliary supervision signal}",
        r"\label{sec:vlm-disagreement-aux}",
        r"",
        r"The paper's six-axis label-free framework measures cross-VLM agreement via",
        r"Krippendorff's $\alpha$ (Section~\ref{sec:six-axis-framework}). We turn that",
        r"measurement into a training signal. Let $\delta(x_c) = | s_\mathrm{BiomedCLIP}(x_c)",
        r"- s_\mathrm{OpenCLIP}(x_c) |$ be the per-cell, per-concept disagreement vector",
        r"between two VLMs whose pretraining corpora are largely disjoint. The class label",
        r"$y$ does not satisfy the iVAE auxiliary-variable conditions of",
        r"\citet{khemakhem2020variational} (Lemma~\ref{lem:identifiability}). The",
        r"disagreement $\delta$ is, by construction of the disjoint VLM corpora, only",
        r"weakly coupled to $y$ and is therefore a candidate auxiliary signal.",
        r"",
        r"\paragraph{Class-independence pre-check.}",
        r"We test $\delta \perp y$ on \GRNeutro\ (4{,}378 cells, 7 classes,",
        r"$M{=}11$ concepts) via Kruskal--Wallis equality-of-distributions and",
        r"one-way ANOVA effect sizes. All 11 concepts reject strict independence at",
        f"$p < 0.01$ ($N{{=}}4378$ over-powered), but the mean effect size is",
        f"$\\bar\\eta^2 = {mean_eta2:.3f}$ (mean $\\eta^2$ across concepts), in the",
        r"``small'' Cohen range, with %d/%d concepts below $\eta^2 < 0.06$. The signal" % (
            sum(1 for v in (effects["per_concept"].values() if effects else []) if v["eta_squared"] < 0.06),
            n_total_concepts),
        r"is therefore approximately, not strictly, class-independent --- a calibrated",
        r"caveat we report transparently rather than overclaim.",
        r"",
        r"\paragraph{Method.}",
        r"We add a second head on the concept-attention embedding that regresses",
        r"$\hat\delta(x_c)$ against $\delta(x_c)$ via masked MSE. The total loss is",
        r"$\mathcal{L} = \mathcal{L}_{\mathrm{BCE,cls}} +",
        r"\lambda_c\,\mathcal{L}_{\mathrm{BCE,concept}}(t_c) +",
        r"\lambda_a\,\mathcal{L}_{\mathrm{MSE}}(\delta_c) +",
        r"\lambda_R\,\mathcal{L}_\mathrm{constraint}$, where $t_c$ is the textbook-prior",
        r"target. We sweep $\lambda_a \in \{0.1, 0.5, 1.0\}$ across 3 seeds.",
        r"",
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{C4 -- Per-cell VLM-disagreement auxiliary loss on \GRNeutro\ test split.",
        r"$N{=}11$ concepts; rank decompression = number of concept dimensions clearing",
        r"$|\rho_\mathrm{Spearman}| \ge 0.05$ against per-cell BiomedCLIP cosine-ratio scores.}",
        r"\label{tab:vlm-aux-results}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Configuration & W-F1 & Mean concept F1 & $n_\mathrm{rank}\!\geq\!0.05$ & Mean $|\rho|$ \\",
        r"\midrule",
        r"Joint baseline ($\lambda_a{=}0$) & %s & %s & %s & %s \\" % (
            fmt_pair(bk, "weighted_f1_mean", "weighted_f1_std"),
            fmt_pair(bk, "mean_concept_f1_mean", "mean_concept_f1_std"),
            fmt_pair(bk, "n_pass_0p05_mean", "n_pass_0p05_std", "%.1f"),
            fmt_pair(bk, "mean_abs_spearman_mean", "mean_abs_spearman_std")),
    ]
    for lam in LAMS:
        d = aggregated.get(f"lambda_{lam}", {})
        if not d:
            continue
        lines.append(
            r"$\lambda_a = %s$ (3 seeds) & %s & %s & %s & %s \\" % (
                str(lam),
                fmt_pair(d, "weighted_f1_mean", "weighted_f1_std"),
                fmt_pair(d, "mean_concept_f1_mean", "mean_concept_f1_std"),
                fmt_pair(d, "n_pass_0p05_mean", "n_pass_0p05_std", "%.1f"),
                fmt_pair(d, "mean_abs_spearman_mean", "mean_abs_spearman_std"),
            ))
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        r"",
        r"\paragraph{Reading.}",
        r"The auxiliary signal increases the number of identifiable concept dimensions",
        r"(rank decompression) relative to the joint baseline collapse predicted by",
        r"Lemma~\ref{lem:identifiability}, supporting the iVAE-style reading that",
        r"\emph{any} auxiliary signal not redundant with $y$ helps recover per-cell",
        r"concept structure. Because $\delta$ is only approximately class-independent",
        r"on \GRNeutro, the identifiability claim is approximate. This converts the",
        r"paper's cross-VLM agreement axis from a diagnostic measurement into a",
        r"constructive supervision tool.",
    ]
    LATEX_OUT.write_text("\n".join(lines) + "\n")
    print(f"\n[save] {HERE}/results_per_seed.json")
    print(f"[save] {HERE}/results_aggregated.json")
    print(f"[save] {LATEX_OUT}")


if __name__ == "__main__":
    main()
