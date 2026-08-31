#!/bin/bash
#SBATCH --job-name=bodzas_eval
#SBATCH --partition=cpu_med
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/workdir/mouaddenn/thesis/ch3_gr_neutro/external_eval/bodzas_slurm-%j.out
#SBATCH --error=/gpfs/workdir/mouaddenn/thesis/ch3_gr_neutro/external_eval/bodzas_slurm-%j.err

set -euo pipefail
WORKDIR=/gpfs/workdir/mouaddenn
ROOT=$WORKDIR/thesis/ch3_gr_neutro
EVAL=$ROOT/external_eval
PY=$WORKDIR/envs/ch3/bin/python
MODEL=$ROOT/outputs/B_kitchen_s42/model.pt
BODZAS=$WORKDIR/data/bodzas
MAP=$WORKDIR/data/bodzas/bodzas_to_grneutro_map.json
OUT=$ROOT/outputs/bodzas_eval

mkdir -p $OUT
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

echo "=== Bodzas external-cohort evaluation (CPU) ==="
hostname
date
cd $EVAL
$PY evaluate_on_bodzas.py \
    --model  $MODEL \
    --bodzas $BODZAS \
    --map    $MAP \
    --out    $OUT \
    --batch_size 32 \
    --num_workers 0 \
    --device cpu

echo "=== DONE ==="
date
