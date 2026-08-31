#!/bin/bash
# Leave-one-class-out OOD probe: re-train the joint kitchen with Döhle held out
# (seed 42, same recipe as r1_joint_const_posw_s42), then run inference on the
# held-out Döhle cells to measure where they get classified.
#
# Matches run_train.sh sbatch directives.
#SBATCH --job-name=joint_no_dohle_s42
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/gpfs/workdir/mouaddenn/thesis/ch3_gr_neutro/logs/%x_%j.out
#SBATCH --error=/gpfs/workdir/mouaddenn/thesis/ch3_gr_neutro/logs/%x_%j.err
set -euo pipefail

WORKDIR=/gpfs/workdir/mouaddenn
PY=$WORKDIR/envs/ch3/bin/python
SRC=$WORKDIR/thesis/ch3_gr_neutro
export HF_HOME=$WORKDIR/tmp/hf-cache
export HF_HUB_CACHE=$WORKDIR/tmp/hf-cache/hub
export TORCH_HOME=$WORKDIR/tmp/torch-hub
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4

cd $SRC
echo "[$(date)] node=$(hostname) job=${SLURM_JOB_ID:-local} tag=joint_no_dohle_s42"
nvidia-smi -L 2>&1 | head -2 || true

TAG=joint_no_dohle_s42
$PY train.py \
    --tag "$TAG" \
    --epochs 40 \
    --seed 42 \
    --lambda_constraint 0.1 \
    --lr_backbone 1e-5 \
    --lr_classifier 1e-4 \
    --lr_adapter 1e-3 \
    --lr_constraint 1e-3 \
    --unfreeze_last_n 6 \
    --batch_size 32 \
    --exclude_classes Dohle

# Standard joint-kitchen post-eval (won't block on dohle_holdout_eval.json,
# which train.py writes itself).
$PY evaluate.py --run_dir "$SRC/outputs/$TAG" || true
echo "[$(date)] done"
