"""Cache frozen-backbone features for the AML Matek (and optionally BMC) dataset.

All paths absolute under /gpfs/workdir/mouaddenn. No internet access assumed at runtime
(compute nodes on Ruche have no internet), so every backbone loads from local weights /
HF cache that were pre-populated on the login node.

Output per (backbone, dataset):
  /gpfs/workdir/mouaddenn/data/<dataset>/features/<backbone>.pt
  with keys:
    features: (N, 1+P, d) float16 — index 0 is the CLS (or GAP for ResNet), 1..P are patch tokens
    labels:   (N,) int64        — class indices matching concept_config.json class order
    grid_hw:  (H, W)            — spatial grid shape for reshaping attention (dynamic, not hardcoded)
    filenames: list[str]         — relative paths for reproducibility
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

WORKDIR = Path("/gpfs/workdir/mouaddenn")
WEIGHTS = WORKDIR / "weights"
DATA = WORKDIR / "data"
THESIS = WORKDIR / "thesis" / "aml_matek"

# Redirect HF / torch caches to workdir (shared GPFS, visible to compute nodes).
os.environ.setdefault("HF_HOME", str(WORKDIR / "tmp" / "hf-cache"))
os.environ.setdefault("HF_HUB_CACHE", str(WORKDIR / "tmp" / "hf-cache" / "hub"))
os.environ.setdefault("TORCH_HOME", str(WORKDIR / "tmp" / "torch-hub"))
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# Matek folder abbreviation -> index in concept_config.json "aml_matek_classes".
# Classes not present in Matek (plasma_cell, rbc_platelet, artifact) remain empty.
# LYA (atypical lymphocyte) and MOB (monoblast) map to rare_atypical (13).
# KSC (smudge / Kernschatten) maps to artifact (14) — damaged nuclei, no real cell.
# PMB (bilobed promyelocyte) and PMO both map to promyelocyte (1).
MATEK_FOLDER_TO_CLASS = {
    "MYB": 0,   # myeloblast
    "PMO": 1,   # promyelocyte
    "PMB": 1,   # bilobed promyelocyte (still promyelocyte)
    "MYO": 2,   # myelocyte
    "MMZ": 3,   # metamyelocyte
    "NGB": 4,   # band neutrophil
    "NGS": 5,   # segmented neutrophil
    "EOS": 6,   # eosinophil
    "BAS": 7,   # basophil
    "MON": 8,   # monocyte
    "LYT": 9,   # lymphocyte (typical)
    "EBO": 11,  # erythroblast
    "LYA": 13,  # atypical lymphocyte -> rare_atypical
    "MOB": 13,  # monoblast -> rare_atypical
    "KSC": 14,  # smudge cell -> artifact
}


def find_aml_matek_root() -> Path:
    """Locate the AML Matek image root on disk.

    kagglehub extracts to /gpfs/workdir/mouaddenn/tmp/kagglehub/datasets/.../4
    The plan expects /gpfs/workdir/mouaddenn/data/aml_matek with class folders.
    We search both and also allow a symlink.
    """
    candidates = [
        DATA / "aml_matek",
        WORKDIR / "tmp" / "kagglehub" / "datasets" / "walkersneps" / "aml-cytomorphology-lmu",
    ]
    for base in candidates:
        if not base.exists():
            continue
        # Descend until we find a directory containing at least one known abbreviation folder.
        for root, dirs, _ in os.walk(base):
            root_p = Path(root)
            found = [d for d in dirs if d in MATEK_FOLDER_TO_CLASS]
            if len(found) >= 5:
                return root_p
    raise FileNotFoundError(f"AML Matek class folders not found under: {candidates}")


class FolderDataset(Dataset):
    """Loads single-cell images from folder-per-class layout."""

    def __init__(self, root: Path, folder_to_class: dict, transform):
        self.samples = []  # list of (path, class_idx)
        self.transform = transform
        exts = ("*.tif", "*.tiff", "*.bmp", "*.png", "*.jpg", "*.jpeg", "*.TIF", "*.TIFF")
        for folder, cls in folder_to_class.items():
            d = root / folder
            if not d.is_dir():
                continue
            for pat in exts:
                for p in sorted(d.glob(pat)):
                    self.samples.append((str(p), cls))
        if not self.samples:
            raise RuntimeError(f"No images found under {root} with known class folders.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, cls = self.samples[i]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, cls, path


def build_transform():
    # ImageNet stats; the backbones' linear probing pipelines all use these.
    return transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ---------------------------------------------------------------------------
# Backbone loaders. Each returns a callable that consumes a (B,3,224,224) tensor
# and produces (features, grid_hw) where features is (B, 1+P, d).
# ---------------------------------------------------------------------------
def load_dinobloom_s(device):
    import timm
    # With HF_HUB_CACHE set to the populated shared cache, timm will hit the cache
    # without network access. img_size=224 triggers positional-embedding interpolation.
    model = timm.create_model(
        "hf-hub:1aurent/vit_small_patch14_224.dinobloom",
        pretrained=True,
        img_size=224,
    )
    model.eval().to(device)

    def fwd(x):
        # timm ViT forward_features returns (B, 1+P, d) when global_pool is not applied.
        feats = model.forward_features(x)
        # feats shape: (B, 1+P, d)
        B, T, D = feats.shape
        P = T - 1
        side = int(round(P ** 0.5))
        assert side * side == P, f"non-square patch grid: P={P}"
        return feats, (side, side)

    return fwd, model.embed_dim


def load_dinov2_vitb14(device):
    # Build architecture from the torch-hub cached source (no internet needed),
    # then load the saved state dict.
    hub_dir = Path(os.environ["TORCH_HOME"]) / "facebookresearch_dinov2_main"
    if not hub_dir.exists():
        # Fallback to hub subdir variant
        alt = Path(os.environ["TORCH_HOME"]) / "hub" / "facebookresearch_dinov2_main"
        if alt.exists():
            hub_dir = alt
        else:
            raise FileNotFoundError(f"DINOv2 hub source not found at {hub_dir}")

    # Use torch.hub 'local' source so no network is required.
    sd_path = WEIGHTS / "dinov2_vitb14.pth"
    model = torch.hub.load(str(hub_dir), "dinov2_vitb14", source="local", pretrained=False)
    state = torch.load(sd_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval().to(device)

    def fwd(x):
        # DINOv2 exposes forward_features returning a dict.
        out = model.forward_features(x)
        cls = out["x_norm_clstoken"]           # (B, d)
        patch = out["x_norm_patchtokens"]      # (B, P, d)
        feats = torch.cat([cls.unsqueeze(1), patch], dim=1)  # (B, 1+P, d)
        P = patch.shape[1]
        side = int(round(P ** 0.5))
        assert side * side == P, f"non-square patch grid: P={P}"
        return feats, (side, side)

    return fwd, model.embed_dim


def load_resnet50(device):
    import torchvision.models as tvm
    model = tvm.resnet50(weights=None)
    state = torch.load(WEIGHTS / "resnet50.pth", map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval().to(device)

    # We want the layer4 output (B, 2048, 7, 7) for spatial features,
    # plus a global summary vector as the synthetic "CLS" at index 0.
    features_container = {}

    def hook(_, __, out):
        features_container["layer4"] = out

    model.layer4.register_forward_hook(hook)

    def fwd(x):
        _ = model(x)
        spat = features_container["layer4"]            # (B, C, H, W)
        B, C, H, W = spat.shape
        patch = spat.flatten(2).transpose(1, 2)        # (B, H*W, C)
        gap = patch.mean(dim=1, keepdim=True)          # (B, 1, C) — synthetic CLS
        feats = torch.cat([gap, patch], dim=1)         # (B, 1+HW, C)
        return feats, (H, W)

    return fwd, 2048


def load_backbone(name, device):
    if name == "dinobloom_s":
        return load_dinobloom_s(device)
    if name == "dinov2_vitb14":
        return load_dinov2_vitb14(device)
    if name == "resnet50":
        return load_resnet50(device)
    raise ValueError(f"Unknown backbone: {name}")


def cache_aml_matek(backbone_name, out_path, batch_size, device):
    root = find_aml_matek_root()
    print(f"[matek] root = {root}")
    ds = FolderDataset(root, MATEK_FOLDER_TO_CLASS, build_transform())
    print(f"[matek] {len(ds)} images across {len(set(c for _, c in ds.samples))} class indices")
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    fwd, embed_dim = load_backbone(backbone_name, device)

    all_feats, all_labels, all_paths = [], [], []
    grid_hw = None

    with torch.no_grad():
        for x, y, paths in tqdm(dl, desc=f"cache {backbone_name}"):
            x = x.to(device, non_blocking=True)
            feats, ghw = fwd(x)
            if grid_hw is None:
                grid_hw = ghw
                print(f"[matek] feature shape per image: (1+{feats.shape[1]-1}, {feats.shape[2]}); grid={ghw}")
            all_feats.append(feats.cpu().to(torch.float16))
            all_labels.append(y.clone())
            all_paths.extend(paths)

    features = torch.cat(all_feats, dim=0)     # (N, 1+P, d) float16
    labels = torch.cat(all_labels, dim=0)       # (N,) int64

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "features": features,
        "labels": labels,
        "grid_hw": grid_hw,
        "filenames": all_paths,
        "backbone": backbone_name,
        "dataset": "aml_matek",
        "embed_dim": embed_dim,
        "num_patches": features.shape[1] - 1,
        "num_classes": 15,
    }, out_path)
    size_gb = out_path.stat().st_size / (1024 ** 3)
    print(f"[matek] saved {features.shape} to {out_path} ({size_gb:.2f} GB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=["dinobloom_s", "dinov2_vitb14", "resnet50"])
    ap.add_argument("--dataset", required=True, choices=["aml_matek", "bmc"])
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[main] device={device}; backbone={args.backbone}; dataset={args.dataset}")

    out_path = DATA / args.dataset / "features" / f"{args.backbone}.pt"
    if out_path.exists():
        print(f"[main] {out_path} already exists — skipping.")
        return

    if args.dataset == "aml_matek":
        cache_aml_matek(args.backbone, out_path, args.batch_size, device)
    else:
        raise NotImplementedError("BMC caching not needed for the listed 6 runs.")


if __name__ == "__main__":
    main()
