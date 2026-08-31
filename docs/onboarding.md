# Project overview

This repository contains work on biologically-constrained classification of
cellular images. It is organized into two layers — the MIDL 2025 paper
codebase and a concept-bottleneck extension on AML Matek — and the
documentation under `docs/` is the intended entry point for anyone extending
it.

## The two layers

### MIDL 2025 paper — `src/`
A multi-label classifier on top of DinoBloom-S. Its core is a learnable
constraint matrix $R$ regularized toward a hand-crafted prior $C$ via
$\lVert RR^\top - C\rVert_F^2 + \alpha\lVert R\rVert_1$, combined with a
direct mutex co-activation penalty on the sigmoid outputs, MC-dropout
uncertainty, and adaptive per-class thresholds. Evaluated on GR-Neutro
(7 classes, multi-label), AML Matek (15 classes, single-label), and BMC
(21 classes). The top-level `README.md` has the architecture diagram and the
paper citation.

### AML Matek concept-bottleneck experiments — `aml_matek/`
An extension that places an 18-concept intermediate representation between
the frozen backbone and the classifier. The concepts are hematological
morphology features (nuclear shape, chromatin pattern, granule type, N:C
ratio, cytoplasm staining) derived from Hoffbrand's *Essential Haematology*
and Briggs' *Haematology in Practice*. Full results are in `figures/` and
summarized in `aml_matek/README.md`.

Key observations from this extension:

- **On AML Matek, the constraint-loss weight $\lambda$ has minimal effect.**
  Sweeping $\lambda$ over two orders of magnitude changes class accuracy by
  less than 0.15 pp while the concept-level violation rate drops 7.5×.
  Because the task is 15-class single-label, a softmax output already
  satisfies class-level mutex pairs almost by construction; the constraint
  term is a tight regularizer rather than a headline driver.

- **Rare concepts need class-weighted BCE to avoid collapse.** With only 65
  positive training samples for `band_nucleus` against 5,400 for the
  visually similar `multilobed_nucleus`, shared BCE drives the rare
  concept's F1 to zero. The class-weighted BCE option
  (`--concept_pos_weight` in `aml_matek/train.py`) recovers it at a modest
  cost to common concepts.

## Suggested reading order

1. This file.
2. `docs/concept_design.md` — how to design a concept vocabulary and turn
   clinical knowledge into a constraint matrix.
3. `aml_matek/README.md` — index of every file in `aml_matek/`.
4. `aml_matek/concept_config.json` — a complete concept vocabulary, useful as
   a format reference.
5. `aml_matek/models.py` — reference implementation of the concept-bottleneck
   CBM.
6. `src/models/losses.py` and `src/models/constraint_module.py` — the MIDL
   2025 implementation.

## Practical notes

- Constraint matrices that look reasonable on paper can mask silent failures.
  The `band_nucleus` F1=0 failure in `aml_matek/` (documented in
  `aml_matek/diag_band.py`) is the canonical example: a concept with only 65
  positive training samples collapsed under shared BCE against a visually
  similar concept with ~5,400 positives. Per-concept supports and per-pair
  violation rates should be checked, not just aggregates. The class-weighted
  BCE (`pos_weight = #neg / #pos`) accessible via `--concept_pos_weight`
  in `aml_matek/train.py` recovers rare concepts at a small cost to common
  ones.
- Feature caches from DinoBloom occupy ≈4–7 GB per dataset at fp16. A GPU
  node with ≥32 GB RAM is needed for training; CPU-only training is not
  practical.
- On HPC environments with small home-directory quotas and compute nodes
  without internet, weights, datasets, and environments should live on
  shared scratch (e.g. `/gpfs/workdir/<user>/`), with model weights
  downloaded on the login node before SLURM submission. The path-setup
  pattern in `aml_matek/slurm/run_all.sh` captures this.

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

The DinoBloom-S, DINOv2-ViT-B/14, and ResNet-50 checkpoints used by
`aml_matek/` are fetched once on the login node before training (see
`aml_matek/README.md` for the exact commands).
