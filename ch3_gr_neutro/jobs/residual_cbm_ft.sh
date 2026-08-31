#!/bin/bash
#SBATCH --job-name=rcbm_ft
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/gpfs/workdir/mouaddenn/thesis/ch3_gr_neutro/logs/rcbm_ft_%j.out
#SBATCH --error=/gpfs/workdir/mouaddenn/thesis/ch3_gr_neutro/logs/rcbm_ft_%j.err

set -e
echo "[$(date)] node=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi -L || true

set +u
source ~/.bashrc
conda activate /gpfs/workdir/mouaddenn/envs/ch3
set -u
export OMP_NUM_THREADS=8

cd /gpfs/workdir/mouaddenn/thesis/ch3_gr_neutro
RUNDIR=/gpfs/workdir/mouaddenn/thesis/ch3_gr_neutro/runs/residual_cbm
mkdir -p "$RUNDIR"
FTNPZ="$RUNDIR/dinobloomb_ft_last4_s0_features.npz"

# 1) cache fine-tuned (last-4) DinoBloom-B CLS features over the full corpus
python cache_ft_features.py \
  --variant dinobloom_b --unfreeze_last_n 4 --epochs 30 --seed 0 \
  --out "$FTNPZ"

# 2) PCBM-h + CEM + controls + faithfulness over the FINE-TUNED features
python residual_cbm.py \
  --features "$FTNPZ" \
  --manifest outputs/p3_pilot/cell_manifest_full_extended.json \
  --morpho   outputs/morphometry_concepts/morphometry_concepts.csv \
  --backbone_json outputs/max_classification/ft_dinobloomb_last4.json \
  --backbone_tag ft_dinobloomb_last4 \
  --out "$RUNDIR/results_ft_last4.json" \
  --d_res 10 --d_res_high 64 --emb 16 --epochs 80

echo "[$(date)] DONE -> $RUNDIR/results_ft_last4.json"
