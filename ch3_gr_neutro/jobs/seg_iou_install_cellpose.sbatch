#!/bin/bash
#SBATCH --job-name=cpose_install
#SBATCH --partition=cpu_short
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/gpfs/workdir/mouaddenn/thesis/outputs/seg_iou/jobs/install_%j.out
#SBATCH --error=/gpfs/workdir/mouaddenn/thesis/outputs/seg_iou/jobs/install_%j.err

set -euo pipefail
echo "HOST=$(hostname) START=$(date)"

module load anaconda3/2023.09-0/none-none

VENV=/gpfs/workdir/mouaddenn/envs/cellpose_iou
BASE_PY=/gpfs/workdir/mouaddenn/envs/ch3/bin/python

# Build an isolated venv from the ch3 python so cellpose deps don't pollute ch3.
if [ ! -d "$VENV" ]; then
  "$BASE_PY" -m venv --system-site-packages "$VENV"
fi
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel setuptools

# PyPI install (GitHub is blocked; PyPI works). CPU torch to keep it light.
python -m pip install "cellpose>=3.0,<4.0"

echo "=== versions ==="
python - <<'PY'
import cellpose, torch, numpy, skimage
print("cellpose", cellpose.version)
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("numpy", numpy.__version__, "skimage", skimage.__version__)
PY

# Pre-download pretrained model weights into a project-local cache so the
# eval job never needs egress. Try nuclei + cyto3 (cyto2 fallback).
export CELLPOSE_LOCAL_MODELS_PATH=/gpfs/workdir/mouaddenn/envs/cellpose_iou/models
mkdir -p "$CELLPOSE_LOCAL_MODELS_PATH"
echo "=== model download (CELLPOSE_LOCAL_MODELS_PATH=$CELLPOSE_LOCAL_MODELS_PATH) ==="
python - <<'PY'
import os, traceback
os.environ.setdefault("CELLPOSE_LOCAL_MODELS_PATH",
                      "/gpfs/workdir/mouaddenn/envs/cellpose_iou/models")
from cellpose import models
ok = {}
for name in ["nuclei", "cyto3", "cyto2", "cyto"]:
    try:
        m = models.CellposeModel(gpu=False, model_type=name) if hasattr(models,"CellposeModel") else models.Cellpose(gpu=False, model_type=name)
        ok[name] = "loaded"
        print("MODEL_OK", name)
    except Exception as e:
        ok[name] = "FAIL: %s" % e
        print("MODEL_FAIL", name, repr(e))
        traceback.print_exc()
import json
print("MODEL_STATUS_JSON", json.dumps(ok))
PY

echo "=== cached model files ==="
ls -laR "$CELLPOSE_LOCAL_MODELS_PATH" 2>/dev/null | head -40
ls -la "$HOME/.cellpose/models" 2>/dev/null | head -40 || true

echo "DONE=$(date)"
