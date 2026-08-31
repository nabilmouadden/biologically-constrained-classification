#!/bin/bash
#SBATCH --job-name=concepts_help_perf
#SBATCH --partition=gpu_test
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/gpfs/workdir/mouaddenn/thesis/ch3_gr_neutro/logs/p3_pilot/%x_%j.out
#SBATCH --error=/gpfs/workdir/mouaddenn/thesis/ch3_gr_neutro/logs/p3_pilot/%x_%j.err

# Test the central thesis: does fusing / co-training the MEASURED morphology
# concepts with the DinoBloom-B backbone make GR-Neutro classification MORE
# accurate than the backbone alone, while staying explainable?
#
# Reuses cached DinoBloom-B features + the morphometry concept CSV + the manifest.
# 4 variants (fusion / multitask / concept-residual / rare-class), 6 seeds,
# paired-bootstrap CIs. PLAIN ACCURACY headline. SLURM only (never login node).
set -euo pipefail
WORKDIR=/gpfs/workdir/mouaddenn
PY=$WORKDIR/envs/ch3/bin/python
cd $WORKDIR/thesis/ch3_gr_neutro
mkdir -p logs/p3_pilot outputs/concepts_help_performance
export OMP_NUM_THREADS=8

echo "[run] training + evaluation"
$PY concepts_help_performance.py
echo "[run] rendering report + figures"
$PY concepts_help_report.py
echo "EXIT_OK"
