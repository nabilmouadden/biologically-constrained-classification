#!/bin/bash
#SBATCH --job-name=grn_ch3
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
echo "[$(date)] node=$(hostname) job=$SLURM_JOB_ID tag=${TAG:-?}"
nvidia-smi -L 2>&1 | head -2 || true

$PY train.py "$@"
$PY evaluate.py --run_dir "$SRC/outputs/${TAG_FOR_EVAL:-$(echo "$@" | sed -n 's/.*--tag \([^ ]*\).*/\1/p')}" || true
echo "[$(date)] done"
