#!/bin/bash
#SBATCH --job-name=lfcbm
#SBATCH --output=/gpfs/workdir/mouaddenn/thesis/ch3_gr_neutro/slurm_logs/lfcbm_%j.out
#SBATCH --error=/gpfs/workdir/mouaddenn/thesis/ch3_gr_neutro/slurm_logs/lfcbm_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

# Label-free CBM (Oikarinen et al., ICLR 2023) baseline on GR-Neutro.
# - Inputs: precomputed BiomedCLIP per-cell concept scores (vlm_grounding stage).
# - Model: sparse (L1) and dense (L2) per-class logistic regression on the
#   11 concept scores, C tuned on the validation split (seed 2024).
# - Outputs: outputs/label_free_cbm/{results.json, concept_weights_per_class.pdf}
#
# Note: no GPU needed; sklearn liblinear is plenty fast on 4378x11.

set -euo pipefail

export WORKDIR=/gpfs/workdir/mouaddenn
export PY=$WORKDIR/envs/ch3/bin/python
export HF_HOME=$WORKDIR/tmp/hf-cache
export HF_HUB_CACHE=$WORKDIR/tmp/hf-cache/hub
export TORCH_HOME=$WORKDIR/tmp/torch-hub
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

cd $WORKDIR/thesis/ch3_gr_neutro
mkdir -p outputs/label_free_cbm slurm_logs

$PY label_free_cbm.py \
    --scores outputs/vlm_grounding/scores/biomedclip_scores.npz \
    --output outputs/label_free_cbm \
    --seed 2024
