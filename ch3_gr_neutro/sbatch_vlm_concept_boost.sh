#!/bin/bash
#SBATCH --job-name=vlm_concept_boost
#SBATCH --partition=cpu_short
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/gpfs/workdir/mouaddenn/thesis/ch3_gr_neutro/logs/p3_pilot/%x_%j.out
#SBATCH --error=/gpfs/workdir/mouaddenn/thesis/ch3_gr_neutro/logs/p3_pilot/%x_%j.err

# Definitive in-distribution test: do HuatuoGPT-Vision per-cell concept scores
# boost GR-Neutro classification over the DinoBloom-B backbone alone?
# SLURM-only (loads features + fits classifiers + bootstrap).
set -euo pipefail
WORKDIR=/gpfs/workdir/mouaddenn
PY=$WORKDIR/envs/ch3/bin/python
cd $WORKDIR/thesis/ch3_gr_neutro
mkdir -p logs/p3_pilot outputs/vlm_concept_boost
export OMP_NUM_THREADS=4
$PY vlm_concept_boost.py
echo "EXIT_OK"
