"""CPU-only sanity check: imports, registry build, alignment, model construction.
Runs on Ruche login node (no GPU). Catches syntax & alignment bugs cheaply."""
import os, sys, json
from pathlib import Path
os.environ["HF_HOME"] = "/gpfs/workdir/mouaddenn/tmp/hf-cache"
os.environ["HF_HUB_CACHE"] = "/gpfs/workdir/mouaddenn/tmp/hf-cache/hub"
os.environ["TORCH_HOME"] = "/gpfs/workdir/mouaddenn/tmp/torch-hub"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

sys.path.insert(0, "/gpfs/workdir/mouaddenn/thesis/ch3_gr_neutro")
sys.path.insert(0, "/gpfs/workdir/mouaddenn/thesis/ch3_gr_neutro/vlm_aux_work")

import csv
import numpy as np
import torch
from train_vlm_disagreement_aux import (load_disagreement_registry, lookup_deltas,
                                          JointModelAux)
from models import build_prior_C
from data import read_annotations

reg, conc = load_disagreement_registry(Path("/gpfs/workdir/mouaddenn/thesis/ch3_gr_neutro/vlm_aux_work"))
print(f"[1] registry: {len(reg)} cells; concepts={len(conc)}")

# Check alignment with annotations.csv basenames.
class_names, rows = read_annotations(
    "/gpfs/workdir/mouaddenn/data/gr_neutro_extended/annotations.csv",
    "/gpfs/workdir/mouaddenn/data/gr_neutro_extended")
print(f"[2] annotations: {len(rows)} rows; classes={class_names}")

bn_list = [r[0] for r in rows]
delta_arr, mask = lookup_deltas(bn_list, reg, 11)
print(f"[3] alignment: {mask.sum()}/{len(mask)} cells matched by basename "
      f"(should be == 4378 for full GR-Neutro)")
if mask.sum() < 4378:
    miss = [b for b, m in zip(bn_list, mask) if not m][:5]
    print(f"[3] first 5 missing basenames: {miss}")
    # Show some keys in registry to debug
    print(f"[3] sample registry keys: {list(reg.keys())[:5]}")

# Construct the model on CPU.
cfg = json.loads(Path("/gpfs/workdir/mouaddenn/thesis/ch3_gr_neutro/concept_config_gr_neutro.json").read_text())
prior_C = build_prior_C(cfg["concepts"], cfg["concept_constraint_matrix"])
print(f"[4] prior_C shape: {prior_C.shape}")

# We won't load the backbone (timm download required); test logits-only path.
# Instead, build the architecture pieces directly.
from models import ConceptAdapter
from train_vlm_disagreement_aux import JointModelAux
# Make a tiny mock backbone for shape test
class MockBackbone(torch.nn.Module):
    embed_dim = 64
    def forward(self, x):
        B = x.size(0)
        return torch.randn(B, 17, 64)  # 1 CLS + 16 patches
mock = MockBackbone()
model = JointModelAux(
    backbone=mock, num_concepts=11, num_classes=7,
    prior_C=prior_C, concept_dim=128, num_heads=4, classifier_dropout=0.5,
    mode="joint",
)
print("[5] JointModelAux constructed OK")

# Forward pass to check shapes
x = torch.randn(4, 3, 224, 224)
cls, con, aux = model(x)
print(f"[6] forward: cls={cls.shape}  con={con.shape}  aux={aux.shape}")
assert cls.shape == (4, 7) and con.shape == (4, 11) and aux.shape == (4, 11)
print("[7] all shape assertions pass")
