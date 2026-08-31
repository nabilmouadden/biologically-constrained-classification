#!/bin/bash
# Multi-cohort iVAE-conditioned CBM, 3 seeds {0, 42, 1337}, gpu partition.
#SBATCH --job-name=mcohort_3s
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --array=0-2
#SBATCH --output=/gpfs/workdir/mouaddenn/thesis/ch3_gr_neutro/logs/%x_%A_%a.out
#SBATCH --error=/gpfs/workdir/mouaddenn/thesis/ch3_gr_neutro/logs/%x_%A_%a.err
set -euo pipefail

WORKDIR=/gpfs/workdir/mouaddenn
PY=$WORKDIR/envs/ch3/bin/python
SRC=$WORKDIR/thesis/ch3_gr_neutro
export HF_HOME=$WORKDIR/tmp/hf-cache
export HF_HUB_CACHE=$WORKDIR/tmp/hf-cache/hub
export TORCH_HOME=$WORKDIR/tmp/torch-hub
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

SEEDS=(0 42 1337)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

cd $SRC
echo "[$(date)] node=$(hostname) job=${SLURM_JOB_ID:-local} seed=$SEED"
nvidia-smi -L 2>&1 | head -2 || true

$PY train_multicohort.py \
    --tag mcohort \
    --seed $SEED \
    --epochs 50 \
    --batch_size 256 \
    --lr 1e-3 \
    --w_class 1.0 \
    --w_concept 2.0 \
    --w_aux 0.5 \
    --concept_target_mode hard

echo "[$(date)] done seed=$SEED"
