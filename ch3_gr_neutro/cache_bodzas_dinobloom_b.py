"""Cache DinoBloom-B CLS-token features for Bodzas-2023 WBC.

Mirrors cache_aml_matek_dinobloom_b.py and uses Bodzas zip layout reading
from external_eval/evaluate_on_bodzas.py.

Output: /gpfs/workdir/mouaddenn/data/bodzas/features/dinobloom_b_cls.npz with
keys: features (N, 768), labels (N,) int (alphabetical class index),
class_names (list), members (filename strings).
"""
from __future__ import annotations
import io
import os
import re
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

WORKDIR = Path("/gpfs/workdir/mouaddenn")
DATA = WORKDIR / "data" / "bodzas"
OUT = DATA / "features" / "dinobloom_b_cls.npz"

os.environ.setdefault("HF_HOME", str(WORKDIR / "tmp" / "hf-cache"))
os.environ.setdefault("HF_HUB_CACHE", str(WORKDIR / "tmp" / "hf-cache" / "hub"))
os.environ.setdefault("TORCH_HOME", str(WORKDIR / "tmp" / "torch-hub"))
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# Borrow class-mapper from evaluate_on_bodzas.
BODZAS_LABEL_ALIASES = {
    "neutrophil_segment":   ["neutrophile_segment", "neutrophil_segment",
                              "neutrophils_segment", "segmented_neutrophil",
                              "neutrophil_segmented", "neutrophil_seg",
                              "segs", "seg"],
    "neutrophil_band":      ["neutrophile_band", "neutrophil_band",
                              "band_neutrophil", "neutrophils_band",
                              "bands", "band"],
    "eosinophil":           ["eosinophile", "eosinophil", "eos", "eo"],
    "basophil":             ["basophile", "basophil", "baso", "ba"],
    "lymphocyte":           ["lymphocyte", "lymph", "ly"],
    "monocyte":             ["monocyte", "mono", "mo"],
    "nucleated_red_blood_cell": ["normoblast", "nrbc", "erythroblast",
                                  "nucleated_red_blood_cell"],
    "myeloblast":           ["myeloblast", "blast_myeloid", "myeloid_blast"],
    "lymphoblast":          ["lymphoblast", "blast_lymphoid", "lymphoid_blast"],
}
IMG_SUFFIXES = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower().strip()).strip("_")


def classify(name: str) -> str | None:
    n = _norm(name)
    for canon, aliases in BODZAS_LABEL_ALIASES.items():
        for a in aliases:
            an = _norm(a)
            if n == an or n.startswith(an + "_") or n.endswith("_" + an):
                return canon
    return None


def is_image(name: str) -> bool:
    if "__MACOSX" in name or Path(name).name.startswith("._"):
        return False
    return Path(name).suffix.lower() in IMG_SUFFIXES


def collect_entries(root: Path):
    """Read Bodzas mega-zip and return [(zip, member, source_class), ...]."""
    entries = []
    unmapped = {}
    zips = sorted(root.glob("*.zip"), key=lambda p: p.stat().st_size, reverse=True)
    if not zips:
        raise FileNotFoundError(f"No zip under {root}")
    zp = zips[0]
    print(f"[collect] mega-zip {zp}", flush=True)
    with zipfile.ZipFile(zp) as zf:
        for info in zf.infolist():
            if info.is_dir() or not is_image(info.filename):
                continue
            parts = [p for p in Path(info.filename).parts
                     if p and p != "__MACOSX" and not p.startswith("._")]
            src = None
            for p in parts[:-1]:
                c = classify(p)
                if c is not None:
                    src = c; break
            if src is None:
                raw = Path(info.filename).parent.name or "ROOT"
                unmapped[raw] = unmapped.get(raw, 0) + 1
                continue
            entries.append((str(zp), info.filename, src))
    print(f"[collect] {len(entries)} cells; unmapped: {unmapped}", flush=True)
    return entries


class ZipDS(Dataset):
    def __init__(self, entries, tfm):
        self.entries = entries
        self.tfm = tfm
        self._zips = {}

    def __len__(self): return len(self.entries)

    def _z(self, p):
        if p not in self._zips: self._zips[p] = zipfile.ZipFile(p, "r")
        return self._zips[p]

    def __getitem__(self, i):
        zp, mem, src = self.entries[i]
        zf = self._z(zp)
        try:
            with zf.open(mem) as f: data = f.read()
            img = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224))
        return self.tfm(img), src, mem


def main():
    import timm
    from torchvision import transforms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[init] device={device}", flush=True)
    model = timm.create_model(
        "hf-hub:1aurent/vit_base_patch14_224.dinobloom",
        pretrained=True, img_size=224, num_classes=0,
    ).to(device).eval()
    cfg = model.default_cfg
    mean = cfg.get("mean", (0.5,) * 3); std = cfg.get("std", (0.5,) * 3)
    tfm = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    entries = collect_entries(DATA)
    # Stable alphabetical class index.
    classes = sorted(set(src for _, _, src in entries))
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    print(f"[classes] {classes}", flush=True)

    ds = ZipDS(entries, tfm)
    loader = DataLoader(ds, batch_size=64, num_workers=4, pin_memory=True)

    feats_list, lbl_list, mem_list = [], [], []
    t0 = time.time()
    with torch.no_grad():
        for bi, (xs, srcs, mems) in enumerate(loader):
            xs = xs.to(device)
            feats = model.forward_features(xs)
            cls = feats[:, 0, :].cpu().to(torch.float32)
            feats_list.append(cls)
            lbl_list.extend([cls_to_idx[s] for s in srcs])
            mem_list.extend(list(mems))
            if bi % 20 == 0:
                print(f"[batch {bi}] {len(mem_list)}/{len(ds)} ({time.time()-t0:.0f}s)", flush=True)

    feats = torch.cat(feats_list, dim=0).numpy()
    labels = np.asarray(lbl_list, dtype=np.int64)
    members = np.asarray(mem_list, dtype=object)
    class_names = np.asarray(classes, dtype=object)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, features=feats, labels=labels, members=members,
             class_names=class_names, backbone="dinobloom_b", embed_dim=768)
    print(f"[save] {OUT}  features.shape={feats.shape}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
