#!/bin/bash
#SBATCH --job-name=rcbm_frozen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=/gpfs/workdir/mouaddenn/thesis/ch3_gr_neutro/logs/rcbm_frozen_%j.out
#SBATCH --error=/gpfs/workdir/mouaddenn/thesis/ch3_gr_neutro/logs/rcbm_frozen_%j.err

set -e
echo "[$(date)] node=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi -L || true

set +u
source ~/.bashrc
conda activate /gpfs/workdir/mouaddenn/envs/ch3
set -u
export OMP_NUM_THREADS=4

cd /gpfs/workdir/mouaddenn/thesis/ch3_gr_neutro
python -c "import torch,sklearn,numpy; print('torch',torch.__version__,'cuda',torch.cuda.is_available())"

RUNDIR=/gpfs/workdir/mouaddenn/thesis/ch3_gr_neutro/runs/residual_cbm
mkdir -p "$RUNDIR"

python residual_cbm.py \
  --features outputs/dinobloom_features.npz \
  --manifest outputs/p3_pilot/cell_manifest_full_extended.json \
  --morpho   outputs/morphometry_concepts/morphometry_concepts.csv \
  --backbone_tag frozen_dinobloomb \
  --out "$RUNDIR/results_frozen.json" \
  --d_res 10 --d_res_high 64 --emb 16 --epochs 80

echo "[$(date)] DONE -> $RUNDIR/results_frozen.json"
