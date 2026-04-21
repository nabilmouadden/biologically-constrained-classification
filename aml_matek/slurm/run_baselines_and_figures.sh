#!/bin/bash
#SBATCH --job-name=amlmatek_baseline
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/workdir/mouaddenn/thesis/aml_matek/logs/baseline_%j.out
#SBATCH --error=/gpfs/workdir/mouaddenn/thesis/aml_matek/logs/baseline_%j.err

set -e

export WORKDIR=/gpfs/workdir/mouaddenn
export PY=$WORKDIR/envs/aml_matek/bin/python
export HF_HOME=$WORKDIR/tmp/hf-cache
export HF_HUB_CACHE=$WORKDIR/tmp/hf-cache/hub
export TRANSFORMERS_CACHE=$WORKDIR/tmp/hf-cache
export TORCH_HOME=$WORKDIR/tmp/torch-hub
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

cd $WORKDIR/thesis/aml_matek

echo "=== BASELINE RUNS (3) ==="
$PY train.py --backbone dinobloom_s   --dataset aml_matek --baseline
$PY train.py --backbone dinov2_vitb14 --dataset aml_matek --baseline
$PY train.py --backbone resnet50      --dataset aml_matek --baseline

echo
echo "=== EVALUATE ALL (non-baseline) RUNS ==="
$PY evaluate.py --all || echo "  (eval may skip baseline dirs — expected)"

echo
echo "=== GENERATE FIGURES ==="
$PY make_figures.py

echo
echo "=== TAR FIGURES ==="
cd $WORKDIR/thesis/aml_matek
tar -czf figures.tar.gz figures/
ls -lh figures.tar.gz

echo "=== DONE ==="
