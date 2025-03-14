Official implementation of the paper "Biologically-Constrained Multi-Label Classification with Learnable Domain Knowledge".

## Overview

This repository contains the implementation of a novel approach for learning and enforcing biological constraints in multi-label classification of neutrophil abnormalities. The method combines learnable constraint matrices with adaptive thresholding, effectively preventing contradictory predictions while maintaining high sensitivity.

The main components include:
1. Feature projection layer
2. Learnable constraint matrix generation
3. Adaptive thresholding mechanism with uncertainty estimation

## Repository Structure

```
├── README.md                     # Repository documentation
├── requirements.txt              # Required Python packages
├── src/                          # Source code
│   ├── __init__.py
│   ├── model/                    # Model architecture files
│   │   ├── __init__.py
│   │   ├── constraint_module.py  # Core constraint module implementation
│   │   ├── adaptive_threshold.py # Adaptive thresholding component
│   │   ├── constraint_priors.py  # Prior constraint matrices for datasets
│   │   └── losses.py             # Loss functions implementation
│   ├── utils/                    # Utility functions
│   │   ├── __init__.py
│   │   ├── visualization.py      # For constraint matrix visualization
│   │   └── uncertainty.py        # Uncertainty estimation utilities
│   └── data/                     # Data loading and processing
│       ├── __init__.py
│       └── datasets.py           # Dataset implementations
├── examples/                     # Example usage scripts
│   ├── train.py                  # Training script
│   ├── inference.py              # Inference script
│   └── generate_constraints.py   # Script to generate constraint matrices
└── configs/                      # Configuration files
    └── example_dataset.yaml      # Example dataset config
```

## Installation

```bash
# Clone the repository
git clone https://github.com/username/biologically-constrained-classification.git
cd biologically-constrained-classification

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Generating Constraint Matrices

Before training, you may want to generate and visualize the biological constraint matrices:

```bash
python examples/generate_constraints.py --dataset example_dataset --output_dir ./configs --visualize
```

This will create both a constraint matrix file and a visualization showing the relationships between different classes.

### Training

To train the model on your dataset:

```bash
python examples/train.py --config configs/example_dataset.yaml --output_dir ./checkpoints
```

### Inference

For inference on test data:

```bash
# Run inference with the best model
python examples/inference.py --config configs/example_dataset.yaml --output_dir ./results
```

Or specify a specific checkpoint:

```bash
python examples/inference.py --config configs/example_dataset.yaml --checkpoint path/to/checkpoint.pth --output_dir ./results
```