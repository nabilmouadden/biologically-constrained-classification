#!/bin/bash
# Submit a lambda sweep at {0, 0.01, 0.05, 0.1, 0.3, 1.0}, single seed=2024,
# reusing the primary R1 hyperparameters. Run this after R1 lands to draw
# the Pareto figure.
set -euo pipefail
cd $(dirname "$0")

for lam in 0.0 0.01 0.05 0.1 0.3 1.0; do
  TAG="ls_lam${lam//./p}_s2024"
  sbatch --job-name="ls_${lam//./p}" --export=ALL,TAG=$TAG run_train.sh \
    --tag "$TAG" --epochs 40 --seed 2024 --lambda_constraint "$lam" \
    --lr_backbone 1e-5 --lr_classifier 1e-4 --lr_adapter 1e-3 --lr_constraint 1e-3 \
    --unfreeze_last_n 6 --batch_size 32
done

squeue -u mouaddenn
