# Project overview

This repository contains work on biologically-constrained classification of
cellular images. It is organized into three layers that correspond to how the
project evolved, and the documentation under `docs/` is the intended entry
point for anyone extending it.

## The three layers

### MIDL 2025 paper — `src/`
A multi-label classifier on top of DinoBloom-S with a **learnable constraint
matrix R** regularized toward a hand-crafted prior C via
`‖RRᵀ − C‖_F² + α‖R‖₁`, plus MC-dropout uncertainty and adaptive
thresholding. Evaluated on three datasets: GR-Neutro (7 classes, multi-label),
AML Matek (15 classes, single-label), BMC (21 classes). The top-level
`README.md` has the architecture diagram and the paper citation.

### AML Matek concept-bottleneck experiments — `aml_matek/`
An extension of the MIDL work to a **concept-bottleneck CBM**. Instead of
placing the constraint matrix on the 15 class outputs, the constraint sits on
an 18-concept intermediate representation derived from Hoffbrand's *Essential
Haematology* and Briggs' *Haematology in Practice*. The 18 concepts are
hematological morphology features (nuclear shape, chromatin pattern, granule
type, N:C ratio, cytoplasm staining). Full results are in `figures/` and
summarized in `aml_matek/README.md`.

Key observations carried forward:

- **On AML Matek, constraint strength barely moves any metric.** A sweep of λ
  over two orders of magnitude changes class accuracy by less than 0.15 pp
  while the violation rate drops 7.5×. The ceiling is already near zero
  because a 15-way softmax forces single-winner predictions; class-level
  mutex pairs are almost satisfied by construction. AML Matek is not the
  sandbox where constraint mechanisms show their value.

- **Rare concepts need class-weighted BCE to avoid collapse.** With only 65
  positive training samples for `band_nucleus` against 5,400 for the visually
  similar `multilobed_nucleus`, shared BCE drives the rare concept's F1 to
  zero. The class-weighted BCE option (`--concept_pos_weight`) recovers it at
  a modest cost to common concepts.

### GR-Neutro follow-up — `gr_neutro/`
A seed directory for a follow-up on GR-Neutro, where mutex pairs
(hyper/hypogranulation, hyper/hyposegmentation, Normal vs. any abnormality)
are hard biological contradictions and the multi-label sigmoid outputs can
genuinely fire simultaneously. The starter kit contains a proposed concept
vocabulary (classes-as-concepts, with a decomposed alternative outlined) and
a workflow document for running the pipeline on the dataset.

## Suggested reading order

1. This file.
2. `docs/concept_design.md` — how to design a concept vocabulary, how to turn
   biology into a constraint matrix, what can go wrong.
3. `docs/gr_neutro_notes.md` — the specific code snippet to port and what to
   measure on GR-Neutro.
4. `aml_matek/README.md` — short index of every file in `aml_matek/`.
5. `aml_matek/concept_config.json` — a complete, battle-tested concept vocabulary,
   useful as a format reference.
6. `aml_matek/models.py` — minimal reference implementation of the full CBM
   (~100 lines).
7. `src/models/losses.py` and `src/models/constraint_module.py` — the MIDL
   2025 implementation.

## Practical notes

- The 18 concepts in `aml_matek/` were designed for AML maturation-stage classes,
  not for neutrophil abnormalities. The vocabulary does not transfer
  directly to GR-Neutro; it serves as a format and methodology reference,
  not a biology reference.
- Constraint matrices that look reasonable on paper can mask silent failures.
  The `band_nucleus` F1=0 failure in aml_matek (documented in `aml_matek/diag_band.py`)
  is the canonical example: a concept with only 65 positive training samples
  collapsed under shared BCE against a visually similar concept with ~5,400
  positives. Per-concept supports and per-pair violation rates should be
  checked, not just aggregates. The class-weighted BCE
  (`pos_weight = #neg / #pos`) accessible via the `--concept_pos_weight`
  flag in `aml_matek/train.py` recovers rare concepts at a small cost to common
  ones.
- Feature caches from DinoBloom occupy ≈4–7 GB per dataset at fp16. A GPU
  node with ≥32 GB RAM is needed for training; CPU-only training is not
  practical.
- On the Ruche cluster (and similar HPC environments), home-directory quotas
  are small and compute nodes have no internet access. All weights,
  datasets, and environments should live on shared scratch (e.g.
  `/gpfs/workdir/<user>/`), and model weights must be downloaded to local
  disk on the login node before SLURM submission. The path-setup pattern in
  `aml_matek/slurm/run_all.sh` captures this.

## Environment

```bash
# On a machine with internet access:
conda create --prefix ./envs/aml_matek python=3.10 -y
conda activate ./envs/aml_matek
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# HF / torch caches off the home directory on HPC (compute nodes have no internet):
export HF_HOME=<scratch>/tmp/hf-cache
export HF_HUB_CACHE=$HF_HOME/hub
export TORCH_HOME=<scratch>/tmp/torch-hub
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
```

The DinoBloom-S, DINOv2-ViT-B/14, and ResNet-50 checkpoints used by aml_matek are
fetched once on the login node before training (see `aml_matek/README.md` for the
exact commands).

## What "done" looks like for GR-Neutro follow-up

A fully filled-out `gr_neutro/` directory with:

- a concept config tuned to GR-Neutro biology (not the AML Matek concepts),
- results from the `configs/gr_neutro.yaml` pipeline reporting per-pair mutex
  violation rate alongside weighted-F1 and per-class F1,
- the same 11-figure set regenerated on GR-Neutro (`aml_matek/figures/` and
  `aml_matek/main_results_table.{csv,tex}` serve as the visual template),
- a short research memo summarizing the findings.
