#!/bin/bash
# Submit C1-LOCO SLURM jobs with cache-dependency chain.
#
# Config A: held_out=0 (GR-Neutro)    train on (AML+MLL+Bod)   needs AML, Bod
# Config B: held_out=1 (AML-Matek)    train on (GR+MLL+Bod)    needs Bod
# Config C: held_out=2 (MLL-23)       train on (GR+AML+Bod)    needs AML, Bod  <- canonical
# Config D: held_out=3 (Bodzas)       train on (GR+AML+MLL)    needs AML
#
# Args:
#   $1 = AML cache job id (or "0" if cache already exists)
#   $2 = Bodzas cache job id (or "0" if cache already exists)
#   $3 = configs (default "A B C D")
#   $4 = seeds   (default "0 42 1337")
set -euo pipefail

AML_JOB=${1:-0}
BOD_JOB=${2:-0}
CONFIGS=${3:-"A B C D"}
SEEDS=${4:-"0 42 1337"}

SRC=/gpfs/workdir/mouaddenn/thesis/ch3_gr_neutro
LOGS=$SRC/logs
mkdir -p $LOGS $SRC/jobs $SRC/outputs/c1_loco

declare -A HELD=( [A]=0 [B]=1 [C]=2 [D]=3 )
# Per-config cache dependencies (space-separated job ids).
declare -A DEPS=( [A]="AML BOD" [B]="BOD" [C]="AML BOD" [D]="AML" )

for cfg in $CONFIGS; do
  held=${HELD[$cfg]}
  # Build dependency string for this config.
  deps=""
  for d in ${DEPS[$cfg]}; do
    case $d in
      AML) [ "$AML_JOB" != "0" ] && deps="$deps,afterok:$AML_JOB" ;;
      BOD) [ "$BOD_JOB" != "0" ] && deps="$deps,afterok:$BOD_JOB" ;;
    esac
  done
  # Strip leading comma.
  deps="${deps#,}"
  DEP_FLAG=""
  if [ -n "$deps" ]; then DEP_FLAG="--dependency=$deps"; fi

  for s in $SEEDS; do
    JOB_NAME=c1loco_${cfg}_s${s}
    cat > $SRC/jobs/${JOB_NAME}.sh <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=${LOGS}/%x_%j.out
#SBATCH --error=${LOGS}/%x_%j.err

set -euo pipefail
WORKDIR=/gpfs/workdir/mouaddenn
PY=\$WORKDIR/envs/ch3/bin/python
export HF_HOME=\$WORKDIR/tmp/hf-cache
export HF_HUB_CACHE=\$WORKDIR/tmp/hf-cache/hub
export TORCH_HOME=\$WORKDIR/tmp/torch-hub
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

cd $SRC
echo "[\$(date)] node=\$(hostname) job=\${SLURM_JOB_ID:-local}"
nvidia-smi -L 2>&1 | head -2 || true

\$PY train_loco.py --tag cfgC1L${cfg} --held_out ${held} --seed ${s}
echo "[\$(date)] done"
EOF
    chmod +x $SRC/jobs/${JOB_NAME}.sh
    if [ -n "$DEP_FLAG" ]; then
      echo "[submit] $JOB_NAME with $DEP_FLAG"
      sbatch $DEP_FLAG $SRC/jobs/${JOB_NAME}.sh
    else
      echo "[submit] $JOB_NAME (no deps)"
      sbatch $SRC/jobs/${JOB_NAME}.sh
    fi
  done
done
