#!/bin/bash
# Submit R1 across seeds {42, 1337} (we already have 2024).
set -euo pipefail
cd $(dirname "$0")

for seed in 42 1337; do
  TAG="r1_joint_const_posw_s${seed}"
  sbatch --job-name="r1_s${seed}" --export=ALL,TAG=$TAG run_train.sh \
    --tag "$TAG" --epochs 40 --seed $seed --lambda_constraint 0.1 \
    --lr_backbone 1e-5 --lr_classifier 1e-4 --lr_adapter 1e-3 --lr_constraint 1e-3 \
    --unfreeze_last_n 6 --batch_size 32
done

squeue -u mouaddenn
