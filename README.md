# Biologically-Constrained Multi-Label Classification with Learnable Domain Knowledge

## Overview

This repository contains the implementation of a novel approach for learning and enforcing biological constraints in multi-label classification of neutrophil abnormalities. The method combines learnable constraint matrices with adaptive thresholding, effectively preventing contradictory predictions while maintaining high sensitivity.

The main components include:
1. Feature projection layer
2. Learnable constraint matrix generation
3. Adaptive thresholding mechanism with uncertainty estimation

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
