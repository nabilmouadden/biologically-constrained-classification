"""Diagnose why band_nucleus concept F1 collapsed to 0."""
import json, torch, numpy as np
from pathlib import Path

CH3 = Path("/gpfs/workdir/mouaddenn/thesis/aml_matek")
cfg = json.loads((CH3 / "concept_config.json").read_text())
concepts = cfg["concepts"]

b = torch.load(CH3 / "outputs/dinobloom_s_aml_matek_joint_const/predictions.pt",
               map_location="cpu", weights_only=False)
test = b["test"]
y = test["labels"].numpy()
cp = test["concept_logits"].sigmoid().numpy()
ct = (test["concept_targets"].numpy() >= 0.5).astype(int)
labels_pred = test["class_logits"].argmax(1).numpy()

C_BAND = concepts.index("band_nucleus")
C_MULT = concepts.index("multilobed_nucleus")
C_ROUND = concepts.index("round_oval_nucleus")
C_INDENT = concepts.index("indented_nucleus")

NGB, NGS = 4, 5  # band_neutrophil, segmented_neutrophil
print(f"NGB (band) test samples: {(y == NGB).sum()}")
print(f"NGS (seg)  test samples: {(y == NGS).sum()}")

mask_ngb = (y == NGB)
print(f"\nGT=band_neutrophil (n={mask_ngb.sum()}):")
print(f"  pred band_nucleus prob       mean={cp[mask_ngb, C_BAND].mean():.3f}  "
      f"median={np.median(cp[mask_ngb, C_BAND]):.3f}  max={cp[mask_ngb, C_BAND].max():.3f}")
print(f"  pred multilobed_nucleus prob mean={cp[mask_ngb, C_MULT].mean():.3f}  "
      f"median={np.median(cp[mask_ngb, C_MULT]):.3f}")
print(f"  pred round_oval_nucleus prob mean={cp[mask_ngb, C_ROUND].mean():.3f}")
print(f"  pred indented_nucleus prob   mean={cp[mask_ngb, C_INDENT].mean():.3f}")
print(f"  classifier accuracy on NGB   {(labels_pred[mask_ngb] == NGB).mean():.3f}")
counts = np.bincount(labels_pred[mask_ngb], minlength=15)
for i, c in enumerate(counts):
    if c > 0:
        print(f"    predicted as class {i} ({cfg['aml_matek_classes'][i]}): {c}")

print(f"\nGlobal training distribution of band_nucleus (from soft labels):")
cls_concept = np.array(cfg["class_to_concept_matrix"]["matrix"])
for i, cl in enumerate(cfg["aml_matek_classes"]):
    if cls_concept[i, C_BAND] > 0:
        print(f"  class {i} {cl}: target={cls_concept[i, C_BAND]}")

# Training support: how many TRAIN samples have band_nucleus=1?
train_idx = b.get("train_idx")
# train_idx isn't in predictions.pt; reconstruct from the full feature cache + seed
import sys
sys.path.insert(0, str(CH3))
from train import stratified_3way_split
feat = torch.load(CH3 / "data/aml_matek/features/dinobloom_s.pt",
                  map_location="cpu", weights_only=False) if False else None
# Cheaper: reload labels via predictions.pt cal+test (we know test+cal indices)
# But train_idx isn't dumped. Use the expected distribution:
# NGB has 109 total samples; at 60/20/20, train=65, cal/test=22 each.
# band_nucleus positives in train = 65 (only NGB triggers this concept).
# vs neutrophilic_granules: MYO (3268) + MMZ (15) + NGB (109) + NGS (8484) ≈ 11876 → ~7126 train positives
# vs multilobed: NGS (8484) + EOS (424) + BAS (79) ≈ 8987 → ~5392 train positives
# So band_nucleus has ~65 positives vs neutrophilic ~7126 vs multilobed ~5392.
print(f"\nExpected concept positives in TRAIN split (60% of total):")
print(f"  band_nucleus       ~65   (from NGB only)")
print(f"  multilobed_nucleus ~5392 (from NGS+EOS+BAS)")
print(f"  neutrophilic_gran  ~7126 (from MYO+MMZ+NGB+NGS)")
print(f"  round_oval_nucleus ~many (all non-granulocytic)")
print()
print(f"Imbalance ratio band vs multilobed: ~83x minority")
