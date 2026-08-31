#!/bin/bash
#SBATCH --job-name=seg_iou_full
#SBATCH --partition=cpu_med
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/gpfs/workdir/mouaddenn/thesis/outputs/seg_iou/jobs/evalfull_%j.out
#SBATCH --error=/gpfs/workdir/mouaddenn/thesis/outputs/seg_iou/jobs/evalfull_%j.err
set -euo pipefail
echo "HOST=$(hostname) START=$(date)"
module load anaconda3/2023.09-0/none-none
source /gpfs/workdir/mouaddenn/envs/cellpose_iou/bin/activate
export CELLPOSE_LOCAL_MODELS_PATH=/gpfs/workdir/mouaddenn/envs/cellpose_iou/models
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
DATA=/gpfs/workdir/mouaddenn/data/gr_neutro_extended_cleaned
OUT=/gpfs/workdir/mouaddenn/thesis/outputs/seg_iou
SRC=/gpfs/workdir/mouaddenn/thesis/ch3_gr_neutro
python "$SRC/seg_iou_eval.py" --data-dir "$DATA" --out-dir "$OUT" \
    --per-class 40 --seed 0 --overlays-per-class 3 \
    --nuc-diameter 90 --cell-diameter 170
echo "DONE=$(date)"
