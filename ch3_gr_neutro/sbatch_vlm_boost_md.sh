#!/bin/bash
#SBATCH --job-name=vlm_boost_md
#SBATCH --partition=cpu_short
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=/gpfs/workdir/mouaddenn/thesis/ch3_gr_neutro/logs/p3_pilot/%x_%j.out
#SBATCH --error=/gpfs/workdir/mouaddenn/thesis/ch3_gr_neutro/logs/p3_pilot/%x_%j.err
set -euo pipefail
cd /gpfs/workdir/mouaddenn/thesis/ch3_gr_neutro
/gpfs/workdir/mouaddenn/envs/ch3/bin/python make_vlm_boost_md.py
echo "EXIT_OK"
