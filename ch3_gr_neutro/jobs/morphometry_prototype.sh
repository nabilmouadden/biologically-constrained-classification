#!/bin/bash
#SBATCH --job-name=morpho_proto
#SBATCH --partition=cpu_short
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=/gpfs/workdir/mouaddenn/thesis/outputs/morphometry_concepts/proto_%j.out
#SBATCH --error=/gpfs/workdir/mouaddenn/thesis/outputs/morphometry_concepts/proto_%j.out

set -e
source ~/.bashrc
conda activate /gpfs/workdir/mouaddenn/envs/ch3

cd /gpfs/workdir/mouaddenn/thesis/ch3_gr_neutro
python morphometry_concepts.py \
  --data-dir /gpfs/workdir/mouaddenn/data/gr_neutro_extended_cleaned \
  --out-dir /gpfs/workdir/mouaddenn/thesis/outputs/morphometry_concepts_proto \
  --limit 140 --n-overlays 14
echo "PROTO DONE"
