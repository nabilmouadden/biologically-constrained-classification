#!/bin/bash
# C12 - Patient x cohort joint auxiliary CBM.
# Three seeds on gpua100 when available; falls back to gpu V100S partition.
# Wall budget per seed ~25 min on A100 / ~50 min on V100S.
#SBATCH --job-name=c12_pat_co
#SBATCH --partition=gpua100,gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
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
export OMP_NUM_THREADS=4

cd $SRC

SEEDS=(0 42 1337)
IDX=${SLURM_ARRAY_TASK_ID:-0}
SEED=${SEEDS[$IDX]}
TAG=c12_patient_cohort

echo "[$(date)] node=$(hostname) job=${SLURM_JOB_ID:-local} array=$IDX seed=$SEED"
nvidia-smi -L 2>&1 | head -2 || true

$PY train_multicohort_c12.py \
  --tag "$TAG" \
  --seed "$SEED" \
  --epochs 40 \
  --batch_size 256 \
  --lr 1e-3 \
  --w_aux_cohort 0.5 \
  --w_aux_patient 0.5 \
  --concept_target_mode hard \
  --out_root "$SRC/outputs/multi_cohort_patient"

echo "[$(date)] done seed=$SEED"
