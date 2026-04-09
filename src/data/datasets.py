import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


def build_transforms(transform_config):
    """
    Build a torchvision transform pipeline from YAML config.

    Args:
        transform_config (list): List of transform dicts from config YAML.

    Returns:
        torchvision.transforms.Compose: Composed transforms.
    """
    transform_list = []
    for t in transform_config:
        t_type = t['type']
        if t_type == 'RandomResizedCrop':
            transform_list.append(transforms.RandomResizedCrop(
                t['size'], scale=tuple(t.get('scale', [0.8, 1.0]))
            ))
        elif t_type == 'RandomHorizontalFlip':
            transform_list.append(transforms.RandomHorizontalFlip(p=t.get('p', 0.5)))
        elif t_type == 'RandomVerticalFlip':
            transform_list.append(transforms.RandomVerticalFlip(p=t.get('p', 0.5)))
        elif t_type == 'ColorJitter':
            transform_list.append(transforms.ColorJitter(
                brightness=t.get('brightness', 0),
                contrast=t.get('contrast', 0),
                saturation=t.get('saturation', 0),
                hue=t.get('hue', 0),
            ))
        elif t_type == 'RandomRotation':
            transform_list.append(transforms.RandomRotation(t['degrees']))
        elif t_type == 'Resize':
            transform_list.append(transforms.Resize(t['size']))
        elif t_type == 'CenterCrop':
            transform_list.append(transforms.CenterCrop(t['size']))
        elif t_type == 'ToTensor':
            transform_list.append(transforms.ToTensor())
        elif t_type == 'Normalize':
            transform_list.append(transforms.Normalize(
                mean=t['mean'], std=t['std']
            ))
        else:
            raise ValueError(f"Unsupported transform type: {t_type}")
    return transforms.Compose(transform_list)


class _BaseMultiLabelDataset(Dataset):
    """
    Base class for multi-label image datasets.

    Expects a directory layout with an ``annotations.csv`` file that has columns:
    ``filename`` and one column per class (0/1 labels).

    Args:
        root_dir (str): Root directory of the dataset split.
        class_names (list[str]): Ordered list of class column names.
        transform: torchvision transform to apply to images.
    """

    def __init__(self, root_dir, class_names, transform=None):
        self.root_dir = root_dir
        self.class_names = class_names
        self.transform = transform

        annotations_path = os.path.join(root_dir, 'annotations.csv')
        if os.path.exists(annotations_path):
            self.annotations = pd.read_csv(annotations_path)
        else:
            raise FileNotFoundError(
                f"annotations.csv not found in {root_dir}"
            )

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        img_path = os.path.join(self.root_dir, row['filename'])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        labels = torch.tensor(
            [float(row[c]) for c in self.class_names], dtype=torch.float32
        )
        return image, labels


class GRNeutroDataset(_BaseMultiLabelDataset):
    """GR-Neutro dataset — 7 classes of neutrophil abnormalities."""

    CLASS_NAMES = [
        'Normal', 'Chromatin', 'Dohle', 'Hypergranulation',
        'Hypersegmentation', 'Hypogranulation', 'Hyposegmentation',
    ]

    def __init__(self, root_dir, transform=None):
        super().__init__(root_dir, self.CLASS_NAMES, transform)


class AMLMatekDataset(_BaseMultiLabelDataset):
    """AML Matek dataset — 15 cell-type classes."""

    CLASS_NAMES = [
        "Myeloblast", "Promyelocyte", "Myelocyte", "Metamyelocyte",
        "Band Neutrophil", "Segmented Neutrophil", "Eosinophil", "Basophil",
        "Monocyte", "Lymphocyte", "Plasma Cell", "Erythroblast",
        "RBC/Platelet", "Rare/Atypical", "Artifact",
    ]

    def __init__(self, root_dir, transform=None):
        super().__init__(root_dir, self.CLASS_NAMES, transform)


class BMCDataset(_BaseMultiLabelDataset):
    """Bone Marrow Cell dataset — 21 cell-type classes."""

    CLASS_NAMES = [
        "Myeloblast", "Promyelocyte", "Myelocyte", "Metamyelocyte",
        "Band Neutrophil", "Segmented Neutrophil", "Eosinophil", "Basophil",
        "Monocyte", "Lymphocyte", "Plasma Cell", "Erythroblast",
        "Megakaryocyte", "Pro-Erythroblast", "Baso-Erythroblast",
        "Poly-Erythroblast", "Ortho-Erythroblast", "RBC",
        "Artifact", "Smudge", "Other",
    ]

    def __init__(self, root_dir, transform=None):
        super().__init__(root_dir, self.CLASS_NAMES, transform)


_DATASET_REGISTRY = {
    'gr_neutro': GRNeutroDataset,
    'aml_matek': AMLMatekDataset,
    'bmc': BMCDataset,
}


def get_dataset(dataset_config, split='train'):
    """
    Factory function to create a dataset from config.

    Args:
        dataset_config (dict): Dataset section of the YAML config. Must contain
            ``name`` and ``root_dir``. May contain ``transform.<split>`` list.
        split (str): One of 'train', 'val', 'test'.

    Returns:
        Dataset: A PyTorch Dataset instance.
    """
    name = dataset_config['name'].lower()
    if name not in _DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {list(_DATASET_REGISTRY.keys())}"
        )

    root_dir = os.path.join(dataset_config['root_dir'], split)

    # Build transforms from config if present
    transform = None
    transform_config = dataset_config.get('transform', {})
    if split in transform_config:
        transform = build_transforms(transform_config[split])

    dataset_cls = _DATASET_REGISTRY[name]
    return dataset_cls(root_dir=root_dir, transform=transform)
