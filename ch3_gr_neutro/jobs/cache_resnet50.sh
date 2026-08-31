#!/bin/bash
#SBATCH --job-name=cache_resnet50
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
# torchvision ResNet50 ImageNet weights download from download.pytorch.org (NOT HF).
# TORCH_HOME points at the workdir cache; do NOT set HF offline flags here so the
# checkpoint can be fetched once (in-scope public model-checkpoint download).
export TORCH_HOME=$WORKDIR/tmp/torch-hub

cd $SRC
echo "[$(date)] node=$(hostname) job=${SLURM_JOB_ID:-local}"
nvidia-smi -L 2>&1 | head -2 || true

$PY cache_barrera_resnet50.py
echo "[$(date)] done"
