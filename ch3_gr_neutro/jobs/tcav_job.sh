#!/bin/bash
#SBATCH --job-name=tcav
#SBATCH --output=/gpfs/workdir/mouaddenn/thesis/ch3_gr_neutro/logs/tcav_%j.out
#SBATCH --error=/gpfs/workdir/mouaddenn/thesis/ch3_gr_neutro/logs/tcav_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
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

cd "$SRC"
echo "[$(date)] node=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi -L 2>&1 | head -2 || true

mkdir -p "$SRC/outputs/tcav"
$PY tcav.py \
  --classifier "$SRC/outputs/vanilla_s42/model.pt" \
  --features   "$SRC/outputs/dinobloom_features.npz" \
  --config     "$SRC/concept_config_gr_neutro.json" \
  --annotations /gpfs/workdir/mouaddenn/data/gr_neutro_extended/annotations.csv \
  --output     "$SRC/outputs/tcav" \
  --bootstrap  50

echo "[$(date)] done"
