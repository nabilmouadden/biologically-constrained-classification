#!/bin/bash
#SBATCH --job-name=amlmatek_lambda
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/gpfs/workdir/mouaddenn/thesis/aml_matek/logs/lambda_%j.out
#SBATCH --error=/gpfs/workdir/mouaddenn/thesis/aml_matek/logs/lambda_%j.err

set -e
export WORKDIR=/gpfs/workdir/mouaddenn
export PY=$WORKDIR/envs/aml_matek/bin/python
export HF_HOME=$WORKDIR/tmp/hf-cache
export HF_HUB_CACHE=$WORKDIR/tmp/hf-cache/hub
export TORCH_HOME=$WORKDIR/tmp/torch-hub
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

cd $WORKDIR/thesis/aml_matek

# DinoBloom-S joint + class-weighted concept BCE, sweep lambda_constraint.
# Each run writes to outputs/lambda_sweep/lam<val>_posw/.
# Note: lam=0 is equivalent to unconstrained (R-loss and violation-loss both zeroed).
for LAM in 0.0 0.01 0.05 0.1 0.3 1.0; do
  # Sanitize lambda into a filesystem-safe tag: 0.05 -> lam0p05_posw
  TAG="lambda_sweep/lam$(echo $LAM | tr '.' 'p')_posw"
  echo "=== λ=$LAM  →  outputs/$TAG ==="
  $PY train.py \
    --backbone dinobloom_s --dataset aml_matek \
    --joint --constrained --concept_pos_weight \
    --lambda_constraint "$LAM" \
    --tag "$TAG"
done

echo
echo "=== REGENERATE FIGURES (includes lambda sweep) ==="
$PY make_figures.py

echo
echo "=== REPACK TAR ==="
tar -czf figures.tar.gz figures/
ls -lh figures.tar.gz

echo "=== DONE ==="
