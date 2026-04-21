# Biologically-Constrained Multi-Label Classification with Learnable Domain Knowledge

[![MIDL 2025](https://img.shields.io/badge/MIDL-2025-blue)](https://2025.midl.io/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-ee4c2c.svg)](https://pytorch.org/)

Official implementation of **"Biologically-Constrained Multi-Label Classification with Learnable Domain Knowledge"**, accepted at **MIDL 2025** (Medical Imaging with Deep Learning).

> **Authors:** Nabil Mouadden, Veronique Verge, Ahmadreza Arbab, Jean-Baptiste Micol, Elsa Bernard, Aline Renneville, Stergios Christodoulidis, Maria Vakalopoulou
>
> **Affiliations:** MICS, CentraleSupelec, Paris-Saclay University | IHU PRISM, Gustave Roussy

## Abstract

Although recent foundation models trained in a self-supervised setting have shown promise in cellular image analysis, they often produce biologically impossible predictions when handling multiple concurrent abnormalities. We present a novel and modular approach to enforce biological constraints in multi-label medical imaging classification. Building on the DinoBloom hematological foundation model, our method combines **learnable constraint matrices** with **adaptive thresholding**, effectively preventing contradictory predictions while maintaining high sensitivity. Extensive experiments on three datasets demonstrate significant improvements over different foundation models and the state-of-the-art methods.

## Architecture

```
Input Image
    │
    ▼
┌──────────────┐
│  DinoBloom-S │  (frozen backbone)
│  Foundation   │
│  Model        │
└──────┬───────┘
       │ features (384-dim)
       ▼
┌──────────────────────────────────┐
│     Constraint Module            │
│                                  │
│  ┌─────────────┐  ┌──────────┐  │
│  │  Feature     │  │  Prior   │  │
│  │  Projection  │  │  Matrix  │  │
│  │  + Attention │  │  C       │  │
│  └──────┬──────┘  └────┬─────┘  │
│         │              │         │
│         ▼              ▼         │
│  ┌─────────────────────────┐    │
│  │  Learnable Constraint   │    │
│  │  Matrix R (data-driven) │    │
│  └────────────┬────────────┘    │
│               │                  │
│  ┌────────────▼────────────┐    │
│  │  MC Dropout Classifier  │    │
│  │  + Uncertainty Est.     │    │
│  └────────────┬────────────┘    │
│               │                  │
│  ┌────────────▼────────────┐    │
│  │  Adaptive Thresholding  │    │
│  │  T(p, u) = α·t + β·u   │    │
│  │           + δ·(1-p)     │    │
│  └─────────────────────────┘    │
└──────────────────────────────────┘
       │
       ▼
  Multi-label predictions
  + Uncertainty estimates
  + Constraint matrix
```

## Key Contributions

1. **Learnable Constraint Satisfaction Module** — Automatically discovers and enforces biological relationships between cell abnormalities while maintaining end-to-end differentiability.

2. **Adaptive Thresholding** — A per-class thresholding mechanism that dynamically adjusts to varying degrees of abnormality manifestation, incorporating both prediction confidence and Monte Carlo uncertainty.

3. **Biologically-Grounded Prior Constraints** — Domain knowledge encoded as prior constraint matrices (e.g., mutual exclusivity between Normal and all abnormalities) that guide learning.

## Supported Datasets

| Dataset | Classes | Description |
|---------|---------|-------------|
| **GR-Neutro** | 7 | Neutrophil abnormalities (Normal, Chromatin, Dohle, Hypergranulation, Hypersegmentation, Hypogranulation, Hyposegmentation) |
| **AML Matek** | 15 | Acute myeloid leukemia cell types |
| **BMC** | 21 | Bone marrow cell morphology |

## Repository layout (post-MIDL additions)

Two directories extend the MIDL 2025 work:

- [`aml_matek/`](./aml_matek/) — concept-bottleneck experiments on AML Matek.
  Introduces an 18-morphological-concept intermediate representation derived
  from Hoffbrand/Briggs, with per-concept mutex and co-occurrence constraints
  and a conformal-coverage evaluation. Reproducible end to end; see
  `aml_matek/README.md`.
- [`gr_neutro/`](./gr_neutro/) — seed for a follow-up on GR-Neutro, including
  a proposed concept vocabulary and a workflow document for extending the
  pipeline.

The intended entry point for anyone extending the project is
[`docs/onboarding.md`](./docs/onboarding.md), which links the three layers
together and suggests a reading order. The methodology guide for designing
concept vocabularies and constraint matrices is at
[`docs/concept_design.md`](./docs/concept_design.md).

## Installation

```bash
git clone https://github.com/nabilmouadden/biologically-constrained-classification.git
cd biologically-constrained-classification

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Generate Constraint Matrices

Generate and visualize the biological constraint matrices encoding domain knowledge:

```bash
python examples/generate_constraints.py --dataset gr_neutro --output_dir ./configs --visualize
```

Available datasets: `gr_neutro`, `aml_matek`, `bmc`, `all`

### 2. Train

```bash
python examples/train.py --config configs/gr_neutro.yaml --output_dir ./checkpoints
```

Key configuration options in `configs/gr_neutro.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model.backbone` | `dinobloom-s` | Backbone architecture |
| `model.dropout_rate` | `0.5` | MC Dropout rate for uncertainty |
| `model.base_threshold` | `0.5` | Base adaptive threshold |
| `training.mc_samples_train` | `5` | MC samples during training |
| `training.mc_samples_val` | `50` | MC samples during validation |
| `training.freeze_backbone` | `true` | Freeze DinoBloom backbone |
| `training.loss.lambda_con` | `0.1` | Constraint-matching loss weight (aligns $R$ with $C$) |
| `training.loss.lambda_viol` | `0.1` | Violation-penalty weight (mutex co-activation at prediction level) |
| `training.loss.lambda_unc` | `0.1` | Uncertainty loss weight |
| `training.loss.lambda_entropy` | `0.01` | Entropy regularization weight |

### 3. Inference

```bash
python examples/inference.py \
    --config configs/gr_neutro.yaml \
    --checkpoint ./checkpoints/best_model.pth \
    --output_dir ./results \
    --mc_samples 50
```

Outputs include:
- Per-class metrics (accuracy, F1, AUC-ROC, uncertainty)
- Calibration curves and ECE/MCE metrics
- Constraint matrix and threshold visualizations
- High-uncertainty misclassification analysis

## Project Structure

```
├── configs/
│   └── gr_neutro.yaml          # MIDL 2025 config for GR-Neutro
├── examples/                   # MIDL 2025 entry-point scripts
│   ├── train.py
│   ├── inference.py
│   └── generate_constraints.py
├── src/                        # MIDL 2025 implementation
│   ├── models/                 # constraint_module, adaptive_threshold, losses, priors
│   ├── data/datasets.py
│   └── utils/                  # uncertainty, visualization
├── aml_matek/                  # Concept-bottleneck experiments on AML Matek
│   ├── concept_config.json     # 18 concepts, class→concept soft matrix, mutex/cooccur list
│   ├── cache_features.py       # One-shot feature extraction (DinoBloom/DINOv2/ResNet)
│   ├── models.py               # ConceptAdapter + ConstraintModule (R-match + viol-loss) + JointModel
│   ├── train.py                # Supports --baseline/joint/frozen × constrained/unconstrained/posw/λ
│   ├── evaluate.py             # Concept F1, violation rate, probe, conformal coverage
│   ├── make_figures.py         # 11 paper figures + LaTeX main table
│   ├── diag_band.py            # Diagnosis script for the band_nucleus F1=0 failure
│   ├── summary_all.py          # Cross-config numerical summary
│   └── slurm/                  # 5 SLURM batch scripts
├── gr_neutro/                  # Starter kit for the GR-Neutro follow-up
│   ├── concept_config.json     # Option A (classes-as-concepts) + Option B (decomposition) TODO
│   └── README.md               # Proposed workflow
├── docs/                       # Methodology and onboarding docs
│   ├── onboarding.md           # Connects MIDL 2025, aml_matek/, and gr_neutro/
│   ├── concept_design.md       # How to design concept vocabularies and constraint matrices
│   └── gr_neutro_notes.md      # What to port from aml_matek/ into MIDL, what to measure
├── figures/                    # Generated figures from the AML Matek experiments
└── requirements.txt
```

## Loss Function

The total loss combines five components:

$$\mathcal{L}_{total} = \mathcal{L}_{BCE} + \lambda_{con}\mathcal{L}_{con} + \lambda_{viol}\mathcal{L}_{viol} + \lambda_{unc}\mathcal{L}_{unc} + \lambda_{ent}\mathcal{L}_{ent}$$

- **BCE Loss** — Binary cross-entropy (summed over K classes, averaged over N samples).
- **Constraint Loss** ($\mathcal{L}_{con} = \\|R R^\top - C\\|_F^2 + \alpha\\|R\\|_1$) — aligns the learned relationship matrix $R$ with the prior $C$.
- **Violation Loss** ($\mathcal{L}_{viol} = \frac{1}{|B|}\sum_{i \in B}\sum_{(a,b)\in\text{mutex}(C)} p_{i,a}\,p_{i,b}$) — direct co-activation penalty over the mutually-exclusive class pairs of $C$, applied to the classifier's MC-averaged sigmoid outputs $p$. Its gradient flows into the classifier, so mutex constraints are enforced at the prediction level rather than only at the level of the side matrix $R$.
- **Uncertainty Loss** — KL divergence + hinge term penalizing high-uncertainty predictions.
- **Entropy Regularization** — Normalized by $1/K^2$ to encourage decisive constraint relationships.

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{mouadden2025biologically,
  title={Biologically-Constrained Multi-Label Classification with Learnable Domain Knowledge},
  author={Mouadden, Nabil and Verge, Veronique and Arbab, Ahmadreza and Micol, Jean-Baptiste and Bernard, Elsa and Renneville, Aline and Christodoulidis, Stergios and Vakalopoulou, Maria},
  booktitle={Medical Imaging with Deep Learning (MIDL)},
  year={2025}
}
```

## License

This work is licensed under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).
