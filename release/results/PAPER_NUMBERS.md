# Reproduce the BJH paper numbers

Every headline number in the paper maps to a shipped artifact in this release
and a command that prints it. Run the one-shot reproducer:

```bash
python results/reproduce_paper.py
```

It reads only the shipped JSONs under `results/acceptance_analyses/` and prints
each paper number next to the value read back from its artifact.

## Provenance / seed note (read this)

The representative seed for the paper's figures and per-class tables is
**cbm_joint s42** (n = 438 held-out test cells, tuned per-class thresholds). The
shipped acceptance-analysis JSONs are that seed. They reproduce the per-class
detection table, its bootstrap CIs, the calibration numbers and the
clinical-utility numbers **exactly**.

The paper's **headline aggregate** macro-F1 0.81 / weighted-F1 0.88 is the
**5-seed mean**. The single representative seed (s42) alone is a little higher
(macro-F1 0.843, weighted-F1 0.900, both derivable from `per_class_cis.json`).
So the shipped single-seed artifacts reproduce the per-class tables and CIs; the
5-seed aggregate is a mean over five such runs.

## Number -> artifact -> command

| Paper number | Value | Shipped artifact | Command |
|---|---|---|---|
| Per-class detection (Table 2): F1 Normal 0.99, Hyperseg 0.91, Hypogran 0.90, Hypergran 0.89, Hyposeg 0.80, Döhle 0.73, Chromatin 0.68 | per-class + 1000× bootstrap 95% CI | `acceptance_analyses/per_class_cis.json` | `python results/reproduce_paper.py` |
| Macro-F1 (5-seed aggregate) | 0.81 | 5-seed mean; single-seed s42 = 0.843 from `per_class_cis.json` | `python results/reproduce_paper.py` |
| Weighted-F1 (5-seed aggregate) | 0.88 | 5-seed mean; single-seed s42 = 0.900 from `per_class_cis.json` | `python results/reproduce_paper.py` |
| Calibration: pooled ECE | 0.054 | `acceptance_analyses/per_class_calibration.json` (`pooled_ece_flat`) | `python results/reproduce_paper.py` |
| Calibration: pooled Brier | 0.029 | `acceptance_analyses/per_class_calibration.json` (`pooled_brier_flat`) | `python results/reproduce_paper.py` |
| Per-class Brier range | 0.007–0.055 | `acceptance_analyses/per_class_calibration.json` (per-class `brier`) | `python results/reproduce_paper.py` |
| Internal abnormal-vs-normal AUROC | 0.997 | `acceptance_analyses/clinical_utility.json` (`abnormal_vs_normal.auroc`) | `python results/reproduce_paper.py` |
| Triage exact-match accuracy: no-defer / 10% / 20% | 0.836 / 0.886 / 0.946 | `acceptance_analyses/clinical_utility.json` (`triage_yield`) | `python results/reproduce_paper.py` |
| External transfer AUROC (Barrera–Merino 2024) | 0.851 ± 0.049 | paper table `m1_external_block.tex`; external cohort not redistributed | — (external cohort not shipped) |
| Interpretability head W-F1 (fine-tuned): backbone 0.925, CEM 0.915, PCBM-h 0.917, pure-CBM 0.883 | 6-seed mean W-F1 | `../weights/dinobloomb_ft_last4_s0_features.npz` + `data/` | `python code/residual_cbm.py --out runs/residual_cbm_ft.json` |
| Joint-CBM per-class F1 (representative seed s42) | Normal 0.972 … Döhle 0.744 | `figures_v2/per_class_breakdown.csv` (`r1v14_kitchen_s42`) | see `accuracy_by_abnormality.md` §1 |

## What recomputes end-to-end vs. what is a cached artifact

- **Recomputes from shipped model + data (CPU):** the interpretability-head W-F1
  table (backbone / CEM / PCBM-h / pure-CBM, frozen and fine-tuned) via
  `python code/residual_cbm.py` on the shipped feature bank + `data/`.
- **Read back from cached JSONs (CPU, instant):** the per-class detection table
  and CIs, the calibration numbers, and the clinical-utility numbers, via
  `python results/reproduce_paper.py`. The JSONs are the outputs of
  `acceptance_analyses/compute_acceptance.py`, which recomputes them from the
  frozen s42 test predictions (three input artifacts named in that script's
  header; not redistributed because they carry per-cell test predictions).
- **Needs the external cohort:** the Barrera–Merino transfer AUROC 0.851 — the
  external cohort is not redistributed.
