"""Cache torchvision ResNet50 (ImageNet-1k) pooled features (2048-d) for the
generic-CNN arm of the CNN-vs-DinoBloom external-generalisation comparison.

Extracts the global-average-pooled 2048-d feature (the layer just before the
1000-way ImageNet fc) for:
  (i)  GR-Neutro extended — ALL cells, read from disk, labelled later by the
       annotations.csv join in the probe (we save `paths` to match the GR
       DinoBloom bank format so the probe joins identically).
  (ii) Barrera-Merino NeuNN — read straight from the single Dataset_NeuNN.zip
       (inode discipline: no loose images extracted), source class from the
       zip folder name, saved as `labels` + `class_names` + `members` to match
       the Barrera DinoBloom bank format.

Fairness: this is the ONLY thing that differs from the DinoBloom arm — a
different frozen encoder. Preprocessing follows the encoder's own canonical
transform (ResNet50 ImageNet: resize 256 -> centre-crop 224 -> ImageNet norm),
exactly as the DinoBloom banks use the DinoBloom canonical transform. Swapping
in each encoder's native preprocessing is part of "swap the encoder", not a
second free variable.

Outputs (match the two DinoBloom bank formats byte-for-byte in key layout):
  GR:      /gpfs/workdir/mouaddenn/thesis/ch3_gr_neutro/outputs/resnet50_features.npz
             keys: features (N,2048), paths (N,), backbone, embed_dim
  Barrera: /gpfs/workdir/mouaddenn/data/barrera_neunn/features/resnet50_pool.npz
             keys: features (5605,2048), labels (N,), class_names, members, backbone, embed_dim
"""
from __future__ import annotations
import io
import os
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

WORKDIR = Path("/gpfs/workdir/mouaddenn")
GR_ROOT = WORKDIR / "data" / "gr_neutro_extended"
GR_OUT = WORKDIR / "thesis" / "ch3_gr_neutro" / "outputs" / "resnet50_features.npz"
BARRERA_DIR = WORKDIR / "data" / "barrera_neunn"
BARRERA_OUT = BARRERA_DIR / "features" / "resnet50_pool.npz"

# ImageNet weights are shipped with torchvision; if the offline torch-hub cache
# has them, use it. TORCH_HOME points at the workdir cache used by other jobs.
os.environ.setdefault("TORCH_HOME", str(WORKDIR / "tmp" / "torch-hub"))

IMG_SUFFIXES = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")


def is_image(name: str) -> bool:
    if "__MACOSX" in name or Path(name).name.startswith("._"):
        return False
    return Path(name).suffix.lower() in IMG_SUFFIXES


def build_transform():
    from torchvision import transforms
    # ResNet50_Weights.IMAGENET1K_V2 canonical transform.
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ----------------------------- GR-Neutro (disk) -----------------------------
class GRDiskDS(Dataset):
    def __init__(self, paths, tfm):
        self.paths = paths
        self.tfm = tfm

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224))
        return self.tfm(img), str(p)


def collect_gr_paths(root: Path):
    idx = {}
    for ext in IMG_SUFFIXES:
        for p in root.rglob(f"*{ext}"):
            if p.name.startswith("._"):
                continue
            idx.setdefault(p.name, p)
    paths = sorted(idx.values(), key=lambda p: p.name)
    print(f"[GR] {len(paths)} images under {root}", flush=True)
    return [str(p) for p in paths]


# ------------------------------ Barrera (zip) -------------------------------
class BarreraZipDS(Dataset):
    def __init__(self, entries, tfm):
        self.entries = entries  # list of (zip_path, member, src_class)
        self.tfm = tfm
        self._zips = {}

    def __len__(self):
        return len(self.entries)

    def _z(self, p):
        if p not in self._zips:
            self._zips[p] = zipfile.ZipFile(p, "r")
        return self._zips[p]

    def __getitem__(self, i):
        zp, mem, src = self.entries[i]
        zf = self._z(zp)
        try:
            with zf.open(mem) as f:
                data = f.read()
            img = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224))
        return self.tfm(img), src, mem


def collect_barrera_entries(root: Path):
    """Dataset_NeuNN/<CLASS>/img_NNNN.jpg — <CLASS> is the source label."""
    zips = sorted(root.glob("*.zip"), key=lambda p: p.stat().st_size, reverse=True)
    if not zips:
        raise FileNotFoundError(f"No zip under {root}")
    zp = zips[0]
    entries = []
    counts = {}
    with zipfile.ZipFile(zp) as zf:
        for info in zf.infolist():
            if info.is_dir() or not is_image(info.filename):
                continue
            parts = [p for p in Path(info.filename).parts
                     if p and p != "__MACOSX" and not p.startswith("._")]
            # class = parent folder name (last dir before the file)
            src = parts[-2] if len(parts) >= 2 else "UNK"
            entries.append((str(zp), info.filename, src))
            counts[src] = counts.get(src, 0) + 1
    print(f"[Barrera] zip {zp.name}: {len(entries)} cells; classes {counts}", flush=True)
    return entries


# --------------------------------- model ------------------------------------
def load_resnet50(device):
    from torchvision.models import resnet50, ResNet50_Weights
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    model.fc = torch.nn.Identity()  # -> 2048-d global-avg-pool feature
    return model.to(device).eval()


def extract(model, loader, device, has_labels: bool):
    feats_list, a_list, b_list = [], [], []
    t0 = time.time()
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if has_labels:
                xs, srcs, mems = batch
                a_list.extend(list(srcs)); b_list.extend(list(mems))
            else:
                xs, paths = batch
                a_list.extend(list(paths))
            xs = xs.to(device)
            f = model(xs).cpu().to(torch.float32)
            feats_list.append(f)
            if bi % 20 == 0:
                print(f"  [batch {bi}] {sum(x.shape[0] for x in feats_list)} "
                      f"({time.time()-t0:.0f}s)", flush=True)
    feats = torch.cat(feats_list, 0).numpy()
    return feats, a_list, b_list


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[init] device={device}", flush=True)
    model = load_resnet50(device)
    tfm = build_transform()

    # ---- GR-Neutro ----
    gr_paths = collect_gr_paths(GR_ROOT)
    gr_loader = DataLoader(GRDiskDS(gr_paths, tfm), batch_size=64,
                           num_workers=4, pin_memory=True)
    gr_feats, gr_pathout, _ = extract(model, gr_loader, device, has_labels=False)
    GR_OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(GR_OUT, features=gr_feats,
             paths=np.asarray(gr_pathout, dtype=object),
             backbone="resnet50_imagenet", embed_dim=2048)
    print(f"[save] {GR_OUT} features={gr_feats.shape}", flush=True)

    # ---- Barrera ----
    entries = collect_barrera_entries(BARRERA_DIR)
    classes = sorted(set(s for _, _, s in entries))
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    b_loader = DataLoader(BarreraZipDS(entries, tfm), batch_size=64,
                          num_workers=4, pin_memory=True)
    b_feats, b_srcs, b_mems = extract(model, b_loader, device, has_labels=True)
    labels = np.asarray([cls_to_idx[s] for s in b_srcs], dtype=np.int64)
    BARRERA_OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(BARRERA_OUT, features=b_feats, labels=labels,
             members=np.asarray(b_mems, dtype=object),
             class_names=np.asarray(classes, dtype=object),
             backbone="resnet50_imagenet", embed_dim=2048)
    print(f"[save] {BARRERA_OUT} features={b_feats.shape} classes={classes}",
          flush=True)

    # Inode discipline: no loose images were written (GR read from existing
    # dataset dir; Barrera read straight from the zip). Nothing to delete.


if __name__ == "__main__":
    main()
