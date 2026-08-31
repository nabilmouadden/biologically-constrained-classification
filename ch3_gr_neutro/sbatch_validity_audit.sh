#!/bin/bash
#SBATCH --job-name=valid_audit
#SBATCH --partition=cpu_short
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=/gpfs/workdir/mouaddenn/thesis/ch3_gr_neutro/runs/validity_audit/valid_audit_%j.out
#SBATCH --error=/gpfs/workdir/mouaddenn/thesis/ch3_gr_neutro/runs/validity_audit/valid_audit_%j.out

set -e
source ~/.bashrc
conda activate /gpfs/workdir/mouaddenn/envs/ch3
export OMP_NUM_THREADS=2

OUT=/gpfs/workdir/mouaddenn/thesis/ch3_gr_neutro/runs/validity_audit
MV2=/gpfs/workdir/mouaddenn/thesis/outputs/morphometry_concepts_v2

cd /gpfs/workdir/mouaddenn/thesis/ch3_gr_neutro
python runs/validity_audit/validity_audit.py \
  --csv "$MV2/morphometry_concepts_v2.csv" \
  --validation "$MV2/validation_v2.json" \
  --out-dir "$OUT"
echo "VALIDITY AUDIT DONE"
