#!/usr/bin/env bash
# Pull all GR-Neutro CBM weights from the Hugging Face Hub into this directory.
# Requires: pip install huggingface_hub   (provides the `hf` CLI)
set -euo pipefail

REPO="nabimu9/gr-neutro-cbm-weights"
DEST="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Downloading $REPO -> $DEST"
if command -v hf >/dev/null 2>&1; then
    hf download "$REPO" --local-dir "$DEST"
else
    python - "$REPO" "$DEST" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], local_dir=sys.argv[2])
PY
fi
echo "Done. Weights are in $DEST"
