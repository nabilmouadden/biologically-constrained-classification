#!/bin/bash
#SBATCH --job-name=seg_iou_smoke
#SBATCH --partition=cpu_short
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --output=/gpfs/workdir/mouaddenn/thesis/outputs/seg_iou/jobs/smoke_%j.out
#SBATCH --error=/gpfs/workdir/mouaddenn/thesis/outputs/seg_iou/jobs/smoke_%j.err

set -euo pipefail
echo "HOST=$(hostname) START=$(date)"
module load anaconda3/2023.09-0/none-none
source /gpfs/workdir/mouaddenn/envs/cellpose_iou/bin/activate
export CELLPOSE_LOCAL_MODELS_PATH=/gpfs/workdir/mouaddenn/envs/cellpose_iou/models
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

DATA=/gpfs/workdir/mouaddenn/data/gr_neutro_extended_cleaned
SMK=/gpfs/workdir/mouaddenn/thesis/outputs/seg_iou/_smoke
SRC=/gpfs/workdir/mouaddenn/thesis/ch3_gr_neutro

python "$SRC/seg_iou_eval.py" --data-dir "$DATA" --out-dir "$SMK" \
    --per-class 2 --seed 1 --overlays-per-class 1 \
    --nuc-diameter 90 --cell-diameter 170

echo "=== per-cell areas/IoU (sanity) ==="
python - <<PY
import pandas as pd
d=pd.read_csv("$SMK/per_cell_iou.csv")
ok=d[d.our_ok==1]
print(ok[["our_nuc_area","cp_nuc_area","cp_nuc_ninst","nuc_iou",
          "our_wbc_area","cp_wbc_area","cp_cell_ninst","wbc_iou"]].to_string())
print("MEAN nuc_iou", round(ok.nuc_iou.mean(),3), "MEAN wbc_iou", round(ok.wbc_iou.mean(),3))
PY
echo "DONE=$(date)"
