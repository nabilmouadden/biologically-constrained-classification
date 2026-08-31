#!/bin/bash
#SBATCH --job-name=calib_rel
#SBATCH --partition=cpu_short
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/gpfs/workdir/mouaddenn/thesis/ch3_gr_neutro/outputs/r1_reannot_mlsplit_b_s42/_calib_rel_%j.out
set -euo pipefail
cd /gpfs/workdir/mouaddenn/thesis/ch3_gr_neutro
/gpfs/workdir/mouaddenn/envs/ch3/bin/python _calib_reliability.py
echo "DONE_CALIB"
