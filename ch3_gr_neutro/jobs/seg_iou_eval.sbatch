#!/bin/bash
#SBATCH --job-name=seg_iou_eval
#SBATCH --partition=cpu_med
#SBATCH --time=03:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/gpfs/workdir/mouaddenn/thesis/outputs/seg_iou/jobs/eval_%j.out
#SBATCH --error=/gpfs/workdir/mouaddenn/thesis/outputs/seg_iou/jobs/eval_%j.err

set -euo pipefail
echo "HOST=$(hostname) START=$(date)"
module load anaconda3/2023.09-0/none-none
source /gpfs/workdir/mouaddenn/envs/cellpose_iou/bin/activate
export CELLPOSE_LOCAL_MODELS_PATH=/gpfs/workdir/mouaddenn/envs/cellpose_iou/models
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

DATA=/gpfs/workdir/mouaddenn/data/gr_neutro_extended_cleaned
OUT=/gpfs/workdir/mouaddenn/thesis/outputs/seg_iou
SRC=/gpfs/workdir/mouaddenn/thesis/ch3_gr_neutro

echo "=== A. IoU vs Cellpose reference (stratified ~40/class) ==="
python "$SRC/seg_iou_eval.py" --data-dir "$DATA" --out-dir "$OUT" \
    --per-class 40 --seed 0 --overlays-per-class 3

echo "=== B. Build manual gold-standard kit (~8/class) ==="
python "$SRC/seg_iou_build_kit.py" --data-dir "$DATA" \
    --out-dir "$OUT/manual_gt_kit" --per-class 8 --seed 7

echo "=== B'. Dry-run the scorer (no human masks yet -> pending) ==="
cp "$SRC/seg_iou_score_kit.py" "$OUT/score_iou.py"
python "$OUT/score_iou.py" --kit-dir "$OUT/manual_gt_kit" || true

echo "DONE=$(date)"
