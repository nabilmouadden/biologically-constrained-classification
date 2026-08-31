#!/bin/bash
#SBATCH --job-name=morpho_v2
#SBATCH --partition=cpu_med
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/gpfs/workdir/mouaddenn/thesis/outputs/morphometry_concepts_v2/full_%j.out
#SBATCH --error=/gpfs/workdir/mouaddenn/thesis/outputs/morphometry_concepts_v2/full_%j.out

set -e
source ~/.bashrc
conda activate /gpfs/workdir/mouaddenn/envs/ch3
export OMP_NUM_THREADS=8

cd /gpfs/workdir/mouaddenn/thesis/ch3_gr_neutro
python morphometry_concepts_v2.py \
  --data-dir /gpfs/workdir/mouaddenn/data/gr_neutro_extended_cleaned \
  --out-dir /gpfs/workdir/mouaddenn/thesis/outputs/morphometry_concepts_v2 \
  --limit 0 --n-overlays 14
echo "V2 FULL DONE"
