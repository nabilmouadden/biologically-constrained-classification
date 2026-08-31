#!/bin/bash
#SBATCH --job-name=faithful_concept_clf
#SBATCH --partition=cpu_short
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/gpfs/workdir/mouaddenn/thesis/ch3_gr_neutro/logs/p3_pilot/%x_%j.out
#SBATCH --error=/gpfs/workdir/mouaddenn/thesis/ch3_gr_neutro/logs/p3_pilot/%x_%j.err

# FAITHFUL-concept classifier: re-ask the questions that failed with unfaithful
# VLM concepts, now with directly-MEASURED morphometry concepts.
# SLURM-only (loads cached DinoBloom features + morphometry CSV, fits classifiers,
# paired bootstrap). Light job -> cpu_short.
set -euo pipefail
WORKDIR=/gpfs/workdir/mouaddenn
PY=$WORKDIR/envs/ch3/bin/python
cd $WORKDIR/thesis/ch3_gr_neutro
mkdir -p logs/p3_pilot outputs/faithful_concept_classifier
export OMP_NUM_THREADS=4
$PY faithful_concept_classifier.py
echo "EXIT_OK"
