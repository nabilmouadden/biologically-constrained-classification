#!/bin/bash
#SBATCH --job-name=amlmatek_posw
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/workdir/mouaddenn/thesis/aml_matek/logs/posw_%j.out
#SBATCH --error=/gpfs/workdir/mouaddenn/thesis/aml_matek/logs/posw_%j.err

set -e
export WORKDIR=/gpfs/workdir/mouaddenn
export PY=$WORKDIR/envs/aml_matek/bin/python
export HF_HOME=$WORKDIR/tmp/hf-cache
export HF_HUB_CACHE=$WORKDIR/tmp/hf-cache/hub
export TORCH_HOME=$WORKDIR/tmp/torch-hub
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

cd $WORKDIR/thesis/aml_matek

# Three pos-weighted runs: DinoBloom (for direct compare to existing joint_const),
# plus DINOv2 and ResNet so the rare-concept fix is evaluated across all backbones.
$PY train.py --backbone dinobloom_s   --dataset aml_matek --joint --constrained --concept_pos_weight
$PY train.py --backbone dinov2_vitb14 --dataset aml_matek --joint --constrained --concept_pos_weight
$PY train.py --backbone resnet50      --dataset aml_matek --joint --constrained --concept_pos_weight

# Re-run eval + figures to include the new results.
$PY evaluate.py --all
$PY make_figures.py
tar -czf figures.tar.gz figures/
ls -lh figures.tar.gz
echo DONE
