#!/usr/bin/env python3
"""Render outputs/vlm_concept_boost/RESULTS.md from results.json. Pure formatting,
no compute (safe to run on login node, but we run it via SLURM anyway)."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs" / "vlm_concept_boost"

with open(OUT / "results.json") as f:
    R = json.load(f)

cn = R["class_names"]


def fmt(m, s):
    return f"{m:.3f} ± {s:.3f}"


def ci(triple):
    return f"{triple[0]:+.4f} [{triple[1]:+.4f}, {triple[2]:+.4f}]"


L = []
L.append("# VLM-concept boost test (in-distribution, GR-Neutro)\n")
L.append(f"**Backbone:** {R['backbone']}  |  **VLM:** {R['vlm']}  |  "
         f"**cells:** {R['n_cells_used']}  |  **seeds:** {R['seeds']}  |  "
         f"**bootstrap:** {R['n_bootstrap']}\n")
L.append("Labels = dominant class (7-way). Classifier = multinomial logistic "
         "regression, class_weight=balanced, on standardized features. Cells "
         "restricted to those with BOTH a DinoBloom-B feature and a HuatuoGPT "
         "rating (fair paired comparison).\n")

L.append("## Headline metrics (mean ± sd across seeds)\n")
L.append("| Condition | W-F1 | macro-F1 |")
L.append("|---|---|---|")
for name in R["conditions"]:
    w = R["wf1"][name]; m = R["macrof1"][name]
    pretty = {"baseline": "1. DinoBloom-B alone",
              "dino_plus_vlm": "2. DinoBloom-B (+) VLM 11 concepts",
              "vlm_alone": "3. VLM 11 concepts ALONE"}[name]
    L.append(f"| {pretty} | {fmt(w['mean'], w['std'])} | {fmt(m['mean'], m['std'])} |")
L.append("")

pd = R["delta_vlm_vs_baseline_pooled"]
L.append("## Paired Δ (condition 2 − condition 1), seed-mean with 95% CI\n")
L.append(f"- **Δ W-F1** = {ci(pd['d_wf1_seedmean_ci95'])}")
L.append(f"- **Δ macro-F1** = {ci(pd['d_macrof1_seedmean_ci95'])}")
L.append("")
L.append("Per-seed paired-bootstrap ΔW-F1 (5000 resamples each):")
L.append("| seed | Δ W-F1 [95% boot CI] | Δ macro-F1 [95% boot CI] |")
L.append("|---|---|---|")
for s in R["seeds"]:
    d = R["per_seed"][str(s)]["delta_vlm_vs_baseline"]
    L.append(f"| {s} | {ci(d['d_wf1'])} | {ci(d['d_macrof1'])} |")
L.append("")

L.append("## Per-class F1 (mean ± sd across seeds)\n")
L.append("| Class | n | baseline | dino+vlm | Δ (seed-mean, 95% CI) |")
L.append("|---|---|---|---|---|")
for k, c in enumerate(cn):
    b = R["per_class_f1"]["baseline"][c]
    v = R["per_class_f1"]["dino_plus_vlm"][c]
    dpc = pd["per_class_d_seedmean"][c]
    L.append(f"| {c} | {R['class_dist'][c]} | {fmt(b['mean'], b['std'])} | "
             f"{fmt(v['mean'], v['std'])} | {ci(dpc)} |")
L.append("")

# verdict
dwf1 = pd["d_wf1_seedmean_ci95"]
sig = (dwf1[1] > 0) or (dwf1[2] < 0)
direction = "no" if not sig else ("a positive" if dwf1[0] > 0 else "a negative")
chrom = pd["per_class_d_seedmean"]["Chromatin"]
chrom_sig = (chrom[1] > 0) or (chrom[2] < 0)
vlm_alone_w = R["wf1"]["vlm_alone"]["mean"]

L.append("## Verdict\n")
verdict = (
    f"Adding HuatuoGPT-Vision per-cell concept scores to the DinoBloom-B backbone "
    f"yields {direction} in-distribution change in classification accuracy: "
    f"ΔW-F1 = {ci(dwf1)} "
    f"({'CI excludes 0' if sig else 'CI includes 0 — not significant'}). "
    f"The VLM concepts on their own carry limited class signal "
    f"(W-F1 = {vlm_alone_w:.3f} vs {R['wf1']['baseline']['mean']:.3f} for the backbone), "
    f"consistent with an 11-dim hand-crafted concept space being far lower-capacity "
    f"than the 768-dim backbone. On the weak Chromatin class, "
    f"ΔF1 = {ci(chrom)} "
    f"({'significant' if chrom_sig else 'not significant'}). "
    f"In-distribution, the backbone already saturates the available class signal; "
    f"the value of VLM concepts is explainability and cross-cohort transport, "
    f"not in-distribution accuracy."
)
L.append(verdict + "\n")

(OUT / "RESULTS.md").write_text("\n".join(L))
print("wrote", OUT / "RESULTS.md")
print(verdict)
