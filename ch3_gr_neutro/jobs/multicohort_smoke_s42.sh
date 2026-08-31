#!/bin/bash
# Multi-cohort iVAE-conditioned CBM smoke run on gpu_test (1 seed).
# Uses GR-Neutro + MLL-23 cached DinoBloom-B CLS-token features. If AML Matek
# B-features have been cached by the time this job starts, AML cohort is
# included as well (3 cohorts); otherwise the script falls back to 2 cohorts.
#SBATCH --job-name=mcohort_s42
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:55:00
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

cd $SRC
echo "[$(date)] node=$(hostname) job=${SLURM_JOB_ID:-local}"
nvidia-smi -L 2>&1 | head -2 || true

# Smoke: 40 epochs, batch 256, w_aux=0.5
$PY train_multicohort.py \
    --tag mcohort_smoke \
    --seed 42 \
    --epochs 40 \
    --batch_size 256 \
    --lr 1e-3 \
    --w_class 1.0 \
    --w_concept 2.0 \
    --w_aux 0.5 \
    --concept_target_mode hard

echo "[$(date)] done"
