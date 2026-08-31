#!/bin/bash
# R4 robustness check: re-train the joint kitchen on a pseudo-patient split
# (Ward-linkage clustering of DinoBloom features into ~50 pseudo-patients,
# then 80/10/10 split keyed by cluster so no cluster appears in two splits).
# Single seed 42, recipe matched to r1_joint_const_posw_s42.
#SBATCH --job-name=pseudo_patient_s42
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/gpfs/workdir/mouaddenn/thesis/ch3_gr_neutro/logs/%x_%j.out
#SBATCH --error=/gpfs/workdir/mouaddenn/thesis/ch3_gr_neutro/logs/%x_%j.err
set -euo pipefail

WORKDIR=/gpfs/workdir/mouaddenn
PY=$WORKDIR/envs/ch3/bin/python
SRC=$WORKDIR/thesis/ch3_gr_neutro
export HF_HOME=$WORKDIR/tmp/hf-cache
export HF_HUB_CACHE=$WORKDIR/tmp/hf-cache/hub
export TORCH_HOME=$WORKDIR/tmp/torch-hub
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4

cd $SRC
echo "[$(date)] node=$(hostname) job=${SLURM_JOB_ID:-local} tag=pseudo_patient_s42"
nvidia-smi -L 2>&1 | head -2 || true

# Step 1: build the pseudo-patient split (idempotent; rebuilds only if missing).
SPLIT_CSV=$SRC/outputs/pseudo_patient_split_K50.csv
SPLIT_META=$SRC/outputs/pseudo_patient_split_K50.meta.json
if [ ! -f "$SPLIT_CSV" ]; then
    echo "[build] $SPLIT_CSV"
    $PY build_pseudo_patient_split.py \
        --features  $SRC/outputs/dinobloom_features.npz \
        --annotations $WORKDIR/data/gr_neutro_extended/annotations.csv \
        --n_clusters 50 \
        --seed 42 \
        --out_csv  $SPLIT_CSV \
        --out_meta $SPLIT_META
else
    echo "[cache] $SPLIT_CSV already exists; skipping rebuild"
fi
echo "---- split meta ----"
cat $SPLIT_META
echo "---- end split meta ----"

# Step 2: re-train joint kitchen on the pseudo-patient split, recipe matched to
# r1_joint_const_posw_s42 (seed 42, lambda=0.1, lr_backbone=1e-5, lr_classifier=1e-4,
# lr_adapter=1e-3, lr_constraint=1e-3, unfreeze_last_n=6, batch=32, 40 epochs).
TAG=r4_pseudo_patient_K50_s42
$PY train.py \
    --tag "$TAG" \
    --split_csv "$SPLIT_CSV" \
    --epochs 40 \
    --seed 42 \
    --lambda_constraint 0.1 \
    --lr_backbone 1e-5 \
    --lr_classifier 1e-4 \
    --lr_adapter 1e-3 \
    --lr_constraint 1e-3 \
    --unfreeze_last_n 6 \
    --batch_size 32

$PY evaluate.py --run_dir "$SRC/outputs/$TAG" || true
echo "[$(date)] done"
