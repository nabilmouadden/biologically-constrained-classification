#!/bin/bash
#SBATCH --job-name=amlmatek_finish
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/workdir/mouaddenn/thesis/aml_matek/logs/finish_%j.out
#SBATCH --error=/gpfs/workdir/mouaddenn/thesis/aml_matek/logs/finish_%j.err

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

# Clean incomplete output dirs from the OOM'd run so training restarts cleanly.
rm -rf outputs/dinov2_vitb14_aml_matek_baseline outputs/dinov2_vitb14_aml_matek_joint_const

echo "=== FINISH MISSING RUNS (4) ==="
$PY train.py --backbone dinov2_vitb14 --dataset aml_matek --baseline
$PY train.py --backbone dinov2_vitb14 --dataset aml_matek --joint --constrained
$PY train.py --backbone resnet50      --dataset aml_matek --baseline
$PY train.py --backbone resnet50      --dataset aml_matek --joint --constrained

echo
echo "=== EVALUATE ALL ==="
$PY evaluate.py --all

echo
echo "=== GENERATE FIGURES ==="
$PY make_figures.py

echo
echo "=== TAR FIGURES ==="
tar -czf figures.tar.gz figures/
ls -lh figures.tar.gz

echo "=== DONE ==="
