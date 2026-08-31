"""Cache DinoBloom-B CLS-token features for AML Matek 2019.

Mirrors thesis/ch3/cache_features.py::cache_aml_matek but loads DinoBloom-B
(`vit_base_patch14_224.dinobloom`, embed_dim=768) and stores ONLY the CLS
token (per-cell 768-d vector). Output is an .npz to match the GR-Neutro
DinoBloom-B cache format (paths, features).

Used by train_multicohort.py.
"""
from __future__ import annotations
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

WORKDIR = Path("/gpfs/workdir/mouaddenn")
DATA = WORKDIR / "data" / "aml_matek"
OUT = DATA / "features" / "dinobloom_b_cls.npz"

os.environ.setdefault("HF_HOME", str(WORKDIR / "tmp" / "hf-cache"))
os.environ.setdefault("HF_HUB_CACHE", str(WORKDIR / "tmp" / "hf-cache" / "hub"))
os.environ.setdefault("TORCH_HOME", str(WORKDIR / "tmp" / "torch-hub"))
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# Matek folder -> class index (15 classes; reused from thesis/ch3/cache_features.py)
MATEK_FOLDER_TO_CLASS = {
    "MYB": 0, "PMO": 1, "PMB": 1, "MYO": 2, "MMZ": 3, "BAS": 4,
    "EOS": 5, "NGB": 6, "NGS": 7, "MON": 8, "LYT": 9, "LYA": 13,
    "MOB": 13, "EBO": 11, "KSC": 14,
}


def main():
    import timm
    from torchvision import transforms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[init] device={device}", flush=True)

    model = timm.create_model(
        "hf-hub:1aurent/vit_base_patch14_224.dinobloom",
        pretrained=True, img_size=224, num_classes=0,
    ).to(device).eval()
    # Force 224x224 transform to match the model.patch_embed expected size.
    # (model.default_cfg.input_size may show 518 from the pretrained config.)
    h, w = 224, 224
    cfg = model.default_cfg
    mean = cfg.get("mean", (0.5,) * 3)
    std = cfg.get("std", (0.5,) * 3)
    print(f"[dino] embed_dim={model.embed_dim}, input=({h},{w}), mean={mean}, std={std}", flush=True)
    tfm = transforms.Compose([
        transforms.Resize((h, w), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # Walk class folders.
    samples: list[tuple[Path, int]] = []
    for folder, cls in MATEK_FOLDER_TO_CLASS.items():
        d = DATA / folder
        if not d.exists():
            print(f"[warn] missing {d}, skipping", flush=True); continue
        for p in sorted(d.glob("*.tiff")):
            samples.append((p, cls))
        for p in sorted(d.glob("*.tif")):
            samples.append((p, cls))
    print(f"[scan] {len(samples)} cells", flush=True)

    BS = 128
    all_cls, all_lbl, all_paths = [], [], []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(samples), BS):
            batch = samples[i:i+BS]
            xs = torch.stack([tfm(Image.open(p).convert("RGB")) for p, _ in batch]).to(device)
            feats = model.forward_features(xs)  # (B, 1+P, d)
            cls = feats[:, 0, :].cpu().to(torch.float32)
            all_cls.append(cls)
            all_lbl.extend([c for _, c in batch])
            all_paths.extend([str(p) for p, _ in batch])
            if (i // BS) % 20 == 0:
                print(f"[cache] {i+len(batch)}/{len(samples)} ({time.time()-t0:.0f}s)", flush=True)
    feats = torch.cat(all_cls, dim=0).numpy()
    labels = np.asarray(all_lbl, dtype=np.int64)
    paths = np.asarray(all_paths, dtype=object)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, features=feats, labels=labels, paths=paths,
             backbone="dinobloom_b", embed_dim=768)
    print(f"[save] {OUT}  features.shape={feats.shape}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
