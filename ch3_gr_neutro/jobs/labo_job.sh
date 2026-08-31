#!/bin/bash
#SBATCH --job-name=labo
#SBATCH --output=/gpfs/workdir/mouaddenn/thesis/ch3_gr_neutro/slurm_logs/labo_%j.out
#SBATCH --error=/gpfs/workdir/mouaddenn/thesis/ch3_gr_neutro/slurm_logs/labo_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

# LaBo (Yang et al., CVPR 2023, "Language in a Bottle") baseline on GR-Neutro.
# - Inputs: precomputed BiomedCLIP scores_v2 (paired pos/neg cosine-diff channel).
# - Model: per-class L1 logistic with top-K=8 concept pruning per class (LaBo
#   signature step), C and threshold tuned on the validation split (seed 2024).
# - Outputs: outputs/labo/{results.json, concept_selection_per_class.pdf}
#
# Note: CPU-only; sklearn liblinear is plenty fast on 4378x11.

set -euo pipefail

export WORKDIR=/gpfs/workdir/mouaddenn
export PY=$WORKDIR/envs/ch3/bin/python
export HF_HOME=$WORKDIR/tmp/hf-cache
export HF_HUB_CACHE=$WORKDIR/tmp/hf-cache/hub
export TORCH_HOME=$WORKDIR/tmp/torch-hub
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

cd $WORKDIR/thesis/ch3_gr_neutro
mkdir -p outputs/labo slurm_logs

$PY labo.py \
    --scores outputs/vlm_grounding/scores_v2/biomedclip/scores.npz \
    --score_key scores_cosdiff \
    --output outputs/labo \
    --seed 2024 \
    --K_top 8 \
    --lfcbm_results outputs/label_free_cbm/results.json
