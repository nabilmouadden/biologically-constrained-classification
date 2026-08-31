#!/bin/bash
#SBATCH --job-name=probe_cvd
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:55:00
#SBATCH --output=/gpfs/workdir/mouaddenn/thesis/ch3_gr_neutro/logs/%x_%j.out
#SBATCH --error=/gpfs/workdir/mouaddenn/thesis/ch3_gr_neutro/logs/%x_%j.err
set -euo pipefail

WORKDIR=/gpfs/workdir/mouaddenn
PY=$WORKDIR/envs/ch3/bin/python
SRC=$WORKDIR/thesis/ch3_gr_neutro

cd $SRC
echo "[$(date)] node=$(hostname) job=${SLURM_JOB_ID:-local}"

# Shared probe: run once per encoder, then aggregate. Byte-identical head code;
# the encoder feature bank is the ONLY variable.
$PY probe_cnn_vs_dinobloom.py --encoder dinobloom_b
$PY probe_cnn_vs_dinobloom.py --encoder resnet50
$PY probe_cnn_vs_dinobloom.py --aggregate
echo "[$(date)] done"
