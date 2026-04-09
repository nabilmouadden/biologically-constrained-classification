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
| `training.loss.lambda_con` | `0.1` | Constraint loss weight |
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
│   └── gr_neutro.yaml          # Configuration for GR-Neutro dataset
├── examples/
│   ├── train.py                # Training script
│   ├── inference.py            # Inference and evaluation script
│   └── generate_constraints.py # Constraint matrix generation
├── src/
│   ├── constants.py            # Shared constants (EPS)
│   ├── models/
│   │   ├── __init__.py         # create_model() factory
│   │   ├── constraint_module.py # Core constraint satisfaction module
│   │   ├── constraint_priors.py # Biological prior matrices
│   │   ├── adaptive_threshold.py # Adaptive thresholding
│   │   └── losses.py           # Multi-component loss function
│   ├── data/
│   │   ├── __init__.py
│   │   └── datasets.py         # Dataset classes and transforms
│   └── utils/
│       ├── uncertainty.py      # Calibration metrics
│       └── visualization.py    # Training curves, constraint viz
├── tests/                      # Comprehensive test suite (142 tests)
│   ├── test_constraint_module.py
│   ├── test_adaptive_threshold.py
│   ├── test_losses.py
│   ├── test_constraint_priors.py
│   ├── test_model_integration.py
│   ├── test_data.py
│   └── test_config_passthrough.py
└── requirements.txt
```

## Loss Function

The total loss combines four components:

$$\mathcal{L}_{total} = \mathcal{L}_{BCE} + \lambda_{con}\mathcal{L}_{con} + \lambda_{unc}\mathcal{L}_{unc} + \lambda_{ent}\mathcal{L}_{ent}$$

- **BCE Loss** — Binary cross-entropy (summed over K classes, averaged over N samples)
- **Constraint Loss** — Frobenius norm between learned R and prior C matrices
- **Uncertainty Loss** — KL divergence + hinge term penalizing high-uncertainty predictions
- **Entropy Regularization** — Normalized by 1/K² to encourage decisive constraint relationships

## Testing

Run the full test suite (142 tests):

```bash
pytest tests/ -v
```

Tests cover:
- Constraint module forward pass, shapes, ranges, MC dropout behavior
- Adaptive threshold clamping, gradients, formula verification
- Loss function components and mathematical correctness
- Prior constraint matrices (symmetry, values from paper Table 3)
- End-to-end model integration (train step + eval step)
- Dataset classes and transform building
- Config passthrough (no hardcoded values)

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
