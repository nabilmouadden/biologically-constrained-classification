#!/bin/bash
#SBATCH --job-name=vlm_aux
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/workdir/mouaddenn/thesis/ch3_gr_neutro/vlm_aux_work/logs/%x_%j.out
#SBATCH --error=/gpfs/workdir/mouaddenn/thesis/ch3_gr_neutro/vlm_aux_work/logs/%x_%j.err
set -euo pipefail

WORKDIR=/gpfs/workdir/mouaddenn
PY=$WORKDIR/envs/ch3/bin/python
SRC=$WORKDIR/thesis/ch3_gr_neutro
AUX=$SRC/vlm_aux_work
export HF_HOME=$WORKDIR/tmp/hf-cache
export HF_HUB_CACHE=$WORKDIR/tmp/hf-cache/hub
export TORCH_HOME=$WORKDIR/tmp/torch-hub
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4

mkdir -p $AUX/logs $AUX/outputs

cd $SRC
echo "[$(date)] node=$(hostname) job=$SLURM_JOB_ID tag=${TAG:-?}  lam_aux=${LAM:-?}  seed=${SEED:-?}"
nvidia-smi -L 2>&1 | head -2 || true

$PY $AUX/train_vlm_disagreement_aux.py \
    --tag "${TAG}" \
    --lambda_aux "${LAM}" \
    --seed "${SEED}" \
    --out_root "$AUX/outputs" \
    --delta_dir "$AUX"

echo "[$(date)] done"
