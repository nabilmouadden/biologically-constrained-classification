"""GR-Neutro extended-dataset loading + stratified multilabel split.

Re-implements the relevant pieces of thesis/gr_neutro/cache_features.py and
thesis/gr_neutro/run_experiments.py so this directory is self-contained
(no sys.path tricks at import time).
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms


def read_annotations(csv_path: str | Path, data_root: str | Path):
    """CSV header: filename,path,<class1>,<class2>,...  Returns (class_names, rows).

    `rows` is a list of (filename, resolved_path, label_list_of_int). The CSV's
    `path` column is ignored — we re-resolve filenames against `data_root` so
    the dataset is portable across machines.
    """
    data_root = Path(data_root)
    index = {}
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        for p in data_root.rglob(ext):
            index.setdefault(p.name, p)

    rows, missing = [], 0
    with open(csv_path) as f:
        r = csv.reader(f)
        header = next(r)
        class_names = header[2:]
        for row in r:
            filename = row[0]
            label = [int(x) for x in row[2:]]
            real = index.get(filename)
            if real is None:
                missing += 1
                continue
            rows.append((filename, str(real), label))
    if missing:
        raise FileNotFoundError(
            f"{missing}/{missing+len(rows)} basenames in {csv_path} not found under {data_root}"
        )
    return class_names, rows


def stratified_multilabel_split(labels_np: np.ndarray, test_size: float = 0.10,
                                 val_size: float = 0.10, seed: int = 42):
    """80/10/10 split stratified by rarest active class. Identical logic to
    thesis/gr_neutro/run_experiments.py::stratified_multilabel_split.
    """
    from sklearn.model_selection import StratifiedShuffleSplit
    K = labels_np.shape[1]
    class_count = labels_np.sum(axis=0).astype(float)
    class_count[class_count == 0] = labels_np.shape[0]
    strat = np.empty(len(labels_np), dtype=int)
    for i in range(len(labels_np)):
        active = np.where(labels_np[i] == 1)[0]
        strat[i] = int(active[np.argmin(class_count[active])]) if len(active) else -1

    idx_all = np.arange(len(labels_np))
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(sss1.split(idx_all, strat))
    rel = val_size / (1.0 - test_size)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=rel, random_state=seed)
    sub_train, sub_val = next(sss2.split(train_val_idx, strat[train_val_idx]))
    return train_val_idx[sub_train], train_val_idx[sub_val], test_idx


class GRNeutroDataset(Dataset):
    def __init__(self, rows, transform, return_filename: bool = False):
        self.rows = rows
        self.transform = transform
        self.return_filename = return_filename

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        filename, path, label = self.rows[i]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        y = torch.tensor(label, dtype=torch.float32)
        if self.return_filename:
            return img, y, filename
        return img, y


def build_train_transform(strong: bool = True):
    """Strong by default — matches the MIDL best-run config (`strong_aug=True`)."""
    if strong:
        base = [
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
        ]
    else:
        base = [transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC)]
    aug = base + [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(30 if strong else 15),
        transforms.ColorJitter(brightness=0.2 if strong else 0.1,
                                contrast=0.2 if strong else 0.1,
                                saturation=0.2 if strong else 0.1, hue=0.05),
    ]
    if strong:
        aug.append(transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.0))], p=0.3))
    aug += [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    if strong:
        aug.append(transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)))
    return transforms.Compose(aug)


def build_eval_transform():
    return transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def make_balanced_sampler(labels_train_np: np.ndarray) -> WeightedRandomSampler:
    """Per-sample weight = sum of (1/class_freq) over its active classes."""
    class_count = labels_train_np.sum(axis=0).clip(min=1)
    class_weight = 1.0 / class_count
    sample_weight = (labels_train_np * class_weight).sum(axis=1)
    sample_weight[sample_weight == 0] = sample_weight[sample_weight > 0].mean()
    return WeightedRandomSampler(weights=sample_weight,
                                  num_samples=len(sample_weight), replacement=True)
