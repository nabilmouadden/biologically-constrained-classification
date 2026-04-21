# AML Matek — concept-bottleneck experiments

This directory holds the code, configs, and SLURM scripts for concept-bottleneck
experiments on the AML Matek 2019 dataset. It is an **extension of the MIDL 2025
work** (`/src/`): instead of putting the constraint matrix on the class outputs,
this version places it on an intermediate **18-concept bottleneck** derived from
clinical hematology textbooks.

The experiments probe how constraint regularization interacts with single-label
classification. On AML Matek the 15-class softmax already assigns exactly one
label per cell, so mutex pairs at the class-output level are near-satisfied
without any regularizer — the violation rate is ~0.004 at the baseline and
drops to ~0.0006 at λ=1.0, with classification accuracy, concept F1 and macro-F1
all flat across two orders of magnitude of λ. The artefact that comes out of
this work is a **direct co-activation penalty** on mutually-exclusive concept
pairs (see `models.py::ConstraintModule.violation_loss`), which is expected
to carry more weight on multi-label tasks with hard biological contradictions
(e.g. GR-Neutro).

## Files

| File | Purpose |
|------|---------|
| `concept_config.json` | 18-concept vocabulary + 15×18 class→concept soft matrix + mutex/cooccur pair list |
| `cache_features.py` | One-shot feature extraction: loads DinoBloom-S / DINOv2-B / ResNet-50 from local weights, runs 224² input through, saves `(N, 1+P, d)` fp16 tensors |
| `models.py` | `ConceptAdapter` (K learnable queries cross-attending to patch tokens), `ConstraintModule` (R-matching loss **+ direct violation loss**), `JointModel` |
| `train.py` | Training loop. Flags: `--baseline`, `--joint/--frozen`, `--constrained/--unconstrained`, `--concept_pos_weight`, `--lambda_constraint`, `--tag` |
| `evaluate.py` | Per-concept F1/AUROC, violation rate, completeness probe, conformal coverage at α∈{0.01, 0.05, 0.1}, attention overlays |
| `make_figures.py` | 11 paper figures + `main_results_table.{csv,tex}` |
| `diag_band.py` | Diagnosis script that located the `band_nucleus` F1=0 failure (83× class-imbalance in concept positives) |
| `summary_all.py` | Cross-config numerical summary for quick-reading |
| `slurm/` | Five SLURM batch scripts (main 6-run sweep, baselines, partial reruns, pos-weight variants, λ sweep) |

## Expected environment

- Python 3.10
- torch==2.5.1+cu121, torchvision, timm, transformers, scikit-learn, pandas, matplotlib, seaborn, tqdm, kagglehub
- All paths in the scripts are absolute under `/gpfs/workdir/mouaddenn/`. Adapt for your environment by editing `WORKDIR` at the top of `cache_features.py`, `train.py`, `evaluate.py`, `make_figures.py`, and each `slurm/*.sh`.

## End-to-end reproduction

```bash
# 1. prepare features once per (backbone, dataset)
python cache_features.py --backbone dinobloom_s --dataset aml_matek
python cache_features.py --backbone dinov2_vitb14 --dataset aml_matek
python cache_features.py --backbone resnet50 --dataset aml_matek

# 2. run the full ablation (these are the main runs in the memo)
python train.py --backbone dinobloom_s --dataset aml_matek --baseline
python train.py --backbone dinobloom_s --dataset aml_matek --joint --unconstrained
python train.py --backbone dinobloom_s --dataset aml_matek --joint --constrained
python train.py --backbone dinobloom_s --dataset aml_matek --joint --constrained --concept_pos_weight
python train.py --backbone dinobloom_s --dataset aml_matek --frozen --unconstrained
python train.py --backbone dinobloom_s --dataset aml_matek --frozen --constrained

# 3. λ sweep (6 points, produces the Pareto curve)
for LAM in 0.0 0.01 0.05 0.1 0.3 1.0; do
  TAG="lambda_sweep/lam$(echo $LAM | tr '.' 'p')_posw"
  python train.py --backbone dinobloom_s --dataset aml_matek \
      --joint --constrained --concept_pos_weight \
      --lambda_constraint $LAM --tag $TAG
done

# 4. evaluate everything, generate figures
python evaluate.py --all
python make_figures.py
```

## Headline results (all in `/figures/`)

- **Accuracy parity holds on all 3 backbones** (ΔAcc ≤ 0.44 pp vs linear-probe baseline).
- **DinoBloom-S + posw beats the black-box baseline on every classification metric** (acc 0.9515 vs 0.9507, macro-F1 0.620 vs 0.568).
- **Constraint is free** — over 2 orders of magnitude of λ scaling, class accuracy varies <0.15 pp while violation rate drops 7.5×.
- **Completeness probe 0.96 on DinoBloom-S** — the 18-concept bottleneck captures ≥96% of the discriminative signal.
- **Conformal coverage ≥0.99** at α=0.05 across all concepts.
