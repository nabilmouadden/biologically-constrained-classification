#!/bin/bash
#SBATCH --job-name=amlmatek_concept
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/gpfs/workdir/mouaddenn/thesis/aml_matek/logs/amlmatek_%j.out
#SBATCH --error=/gpfs/workdir/mouaddenn/thesis/aml_matek/logs/amlmatek_%j.err

set -e

# ---- environment ----
export WORKDIR=/gpfs/workdir/mouaddenn
export PY=$WORKDIR/envs/aml_matek/bin/python

# Redirect all caches away from the home directory (tiny quota).
export HF_HOME=$WORKDIR/tmp/hf-cache
export HF_HUB_CACHE=$WORKDIR/tmp/hf-cache/hub
export TRANSFORMERS_CACHE=$WORKDIR/tmp/hf-cache
export TORCH_HOME=$WORKDIR/tmp/torch-hub
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export KAGGLEHUB_CACHE=$WORKDIR/tmp/kagglehub

cd $WORKDIR/thesis/aml_matek

echo "=== CUDA check ==="
$PY -c "import torch; print('torch', torch.__version__, 'cuda_available=', torch.cuda.is_available(), 'device=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"

echo
echo "=== PHASE 1: cache features ==="
$PY cache_features.py --backbone dinobloom_s --dataset aml_matek
$PY cache_features.py --backbone dinov2_vitb14 --dataset aml_matek
$PY cache_features.py --backbone resnet50 --dataset aml_matek

echo
echo "=== PHASE 2: train (6 runs) ==="
# Priority 1-4: DinoBloom-S ablations
$PY train.py --backbone dinobloom_s  --dataset aml_matek --joint  --constrained
$PY train.py --backbone dinobloom_s  --dataset aml_matek --joint  --unconstrained
$PY train.py --backbone dinobloom_s  --dataset aml_matek --frozen --constrained
$PY train.py --backbone dinobloom_s  --dataset aml_matek --frozen --unconstrained
# Priority 5-6: multi-backbone
$PY train.py --backbone dinov2_vitb14 --dataset aml_matek --joint --constrained
$PY train.py --backbone resnet50      --dataset aml_matek --joint --constrained

echo
echo "=== PHASE 3: evaluate all ==="
$PY evaluate.py --all

echo "=== DONE ==="
