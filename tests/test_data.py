import os
import pytest
import tempfile
import pandas as pd
import numpy as np
from PIL import Image
from src.data.datasets import (
    GRNeutroDataset,
    AMLMatekDataset,
    BMCDataset,
    get_dataset,
    build_transforms,
)


def _create_dummy_dataset(tmp_dir, class_names, n_samples=5):
    """Create a minimal dataset directory with annotations and dummy images."""
    os.makedirs(tmp_dir, exist_ok=True)
    rows = []
    for i in range(n_samples):
        fname = f"img_{i:04d}.png"
        img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
        img.save(os.path.join(tmp_dir, fname))
        labels = {c: float(np.random.randint(0, 2)) for c in class_names}
        labels['filename'] = fname
        rows.append(labels)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(tmp_dir, 'annotations.csv'), index=False)


@pytest.fixture
def gr_neutro_dir(tmp_path):
    d = str(tmp_path / 'gr_neutro')
    _create_dummy_dataset(d, GRNeutroDataset.CLASS_NAMES, n_samples=3)
    return d


@pytest.fixture
def aml_matek_dir(tmp_path):
    d = str(tmp_path / 'aml_matek')
    _create_dummy_dataset(d, AMLMatekDataset.CLASS_NAMES, n_samples=3)
    return d


@pytest.fixture
def bmc_dir(tmp_path):
    d = str(tmp_path / 'bmc')
    _create_dummy_dataset(d, BMCDataset.CLASS_NAMES, n_samples=3)
    return d


class TestGRNeutroDataset:
    def test_len(self, gr_neutro_dir):
        ds = GRNeutroDataset(gr_neutro_dir)
        assert len(ds) == 3

    def test_getitem_shapes(self, gr_neutro_dir):
        ds = GRNeutroDataset(gr_neutro_dir)
        img, labels = ds[0]
        assert isinstance(img, Image.Image)
        assert labels.shape == (7,)

    def test_with_transforms(self, gr_neutro_dir):
        from torchvision import transforms
        t = transforms.Compose([transforms.Resize(16), transforms.ToTensor()])
        ds = GRNeutroDataset(gr_neutro_dir, transform=t)
        img, labels = ds[0]
        assert img.shape[0] == 3  # channels
        assert labels.shape == (7,)


class TestAMLMatekDataset:
    def test_len(self, aml_matek_dir):
        ds = AMLMatekDataset(aml_matek_dir)
        assert len(ds) == 3

    def test_label_shape(self, aml_matek_dir):
        ds = AMLMatekDataset(aml_matek_dir)
        _, labels = ds[0]
        assert labels.shape == (15,)


class TestBMCDataset:
    def test_len(self, bmc_dir):
        ds = BMCDataset(bmc_dir)
        assert len(ds) == 3

    def test_label_shape(self, bmc_dir):
        ds = BMCDataset(bmc_dir)
        _, labels = ds[0]
        assert labels.shape == (21,)


class TestGetDataset:
    def test_factory(self, tmp_path):
        split_dir = str(tmp_path / 'dataset' / 'train')
        _create_dummy_dataset(split_dir, GRNeutroDataset.CLASS_NAMES, n_samples=2)
        config = {
            'name': 'gr_neutro',
            'root_dir': str(tmp_path / 'dataset'),
        }
        ds = get_dataset(config, split='train')
        assert len(ds) == 2

    def test_unknown_dataset(self):
        with pytest.raises(ValueError):
            get_dataset({'name': 'unknown', 'root_dir': '/tmp'}, split='train')


class TestGetDatasetSplits:
    def test_val_split(self, tmp_path):
        split_dir = str(tmp_path / 'dataset' / 'val')
        _create_dummy_dataset(split_dir, GRNeutroDataset.CLASS_NAMES, n_samples=2)
        config = {'name': 'gr_neutro', 'root_dir': str(tmp_path / 'dataset')}
        ds = get_dataset(config, split='val')
        assert len(ds) == 2

    def test_test_split(self, tmp_path):
        split_dir = str(tmp_path / 'dataset' / 'test')
        _create_dummy_dataset(split_dir, GRNeutroDataset.CLASS_NAMES, n_samples=2)
        config = {'name': 'gr_neutro', 'root_dir': str(tmp_path / 'dataset')}
        ds = get_dataset(config, split='test')
        assert len(ds) == 2

    def test_with_transform_config(self, tmp_path):
        split_dir = str(tmp_path / 'dataset' / 'train')
        _create_dummy_dataset(split_dir, GRNeutroDataset.CLASS_NAMES, n_samples=2)
        config = {
            'name': 'gr_neutro',
            'root_dir': str(tmp_path / 'dataset'),
            'transform': {
                'train': [
                    {'type': 'Resize', 'size': 16},
                    {'type': 'ToTensor'},
                ],
            },
        }
        ds = get_dataset(config, split='train')
        img, labels = ds[0]
        assert img.shape == (3, 16, 16)

    def test_no_transform_for_split(self, tmp_path):
        """If transform config exists but not for this split, no transform applied."""
        split_dir = str(tmp_path / 'dataset' / 'val')
        _create_dummy_dataset(split_dir, GRNeutroDataset.CLASS_NAMES, n_samples=2)
        config = {
            'name': 'gr_neutro',
            'root_dir': str(tmp_path / 'dataset'),
            'transform': {
                'train': [{'type': 'Resize', 'size': 16}, {'type': 'ToTensor'}],
            },
        }
        ds = get_dataset(config, split='val')
        img, labels = ds[0]
        assert isinstance(img, Image.Image)


class TestMissingAnnotations:
    def test_missing_annotations_raises(self, tmp_path):
        d = str(tmp_path / 'empty_ds')
        os.makedirs(d, exist_ok=True)
        with pytest.raises(FileNotFoundError):
            GRNeutroDataset(d)


class TestImageModes:
    def test_rgba_image(self, tmp_path):
        """RGBA images should be converted to RGB."""
        d = str(tmp_path / 'rgba_ds')
        os.makedirs(d, exist_ok=True)
        img = Image.fromarray(np.random.randint(0, 255, (32, 32, 4), dtype=np.uint8), mode='RGBA')
        img.save(os.path.join(d, 'img.png'))
        df = pd.DataFrame([{'filename': 'img.png', **{c: 0.0 for c in GRNeutroDataset.CLASS_NAMES}}])
        df.to_csv(os.path.join(d, 'annotations.csv'), index=False)
        ds = GRNeutroDataset(d)
        img_out, _ = ds[0]
        assert img_out.mode == 'RGB'

    def test_grayscale_image(self, tmp_path):
        """Grayscale images should be converted to RGB."""
        d = str(tmp_path / 'gray_ds')
        os.makedirs(d, exist_ok=True)
        img = Image.fromarray(np.random.randint(0, 255, (32, 32), dtype=np.uint8), mode='L')
        img.save(os.path.join(d, 'img.png'))
        df = pd.DataFrame([{'filename': 'img.png', **{c: 0.0 for c in GRNeutroDataset.CLASS_NAMES}}])
        df.to_csv(os.path.join(d, 'annotations.csv'), index=False)
        ds = GRNeutroDataset(d)
        img_out, _ = ds[0]
        assert img_out.mode == 'RGB'


class TestBuildTransforms:
    def test_basic_pipeline(self):
        config = [
            {'type': 'Resize', 'size': 32},
            {'type': 'ToTensor'},
            {'type': 'Normalize', 'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]},
        ]
        t = build_transforms(config)
        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        out = t(img)
        assert out.shape == (3, 32, 32)

    def test_unsupported_transform(self):
        with pytest.raises(ValueError, match="Unsupported"):
            build_transforms([{'type': 'NonExistent'}])

    def test_random_resized_crop(self):
        config = [{'type': 'RandomResizedCrop', 'size': 16, 'scale': [0.8, 1.0]}, {'type': 'ToTensor'}]
        t = build_transforms(config)
        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        out = t(img)
        assert out.shape == (3, 16, 16)

    def test_random_flips(self):
        config = [
            {'type': 'RandomHorizontalFlip', 'p': 1.0},
            {'type': 'RandomVerticalFlip', 'p': 1.0},
            {'type': 'ToTensor'},
        ]
        t = build_transforms(config)
        img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
        out = t(img)
        assert out.shape == (3, 32, 32)

    def test_color_jitter(self):
        config = [
            {'type': 'ColorJitter', 'brightness': 0.1, 'contrast': 0.1, 'saturation': 0.1, 'hue': 0.05},
            {'type': 'ToTensor'},
        ]
        t = build_transforms(config)
        img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
        out = t(img)
        assert out.shape == (3, 32, 32)

    def test_random_rotation(self):
        config = [{'type': 'RandomRotation', 'degrees': 45}, {'type': 'ToTensor'}]
        t = build_transforms(config)
        img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
        out = t(img)
        assert out.shape == (3, 32, 32)

    def test_center_crop(self):
        config = [{'type': 'CenterCrop', 'size': 16}, {'type': 'ToTensor'}]
        t = build_transforms(config)
        img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
        out = t(img)
        assert out.shape == (3, 16, 16)

    def test_full_training_pipeline(self):
        """Test the full training pipeline from gr_neutro.yaml."""
        config = [
            {'type': 'RandomResizedCrop', 'size': 224, 'scale': [0.8, 1.0]},
            {'type': 'RandomHorizontalFlip', 'p': 0.5},
            {'type': 'RandomVerticalFlip', 'p': 0.5},
            {'type': 'ColorJitter', 'brightness': 0.1, 'contrast': 0.1, 'saturation': 0.1, 'hue': 0.05},
            {'type': 'RandomRotation', 'degrees': 10},
            {'type': 'ToTensor'},
            {'type': 'Normalize', 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]},
        ]
        t = build_transforms(config)
        img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
        out = t(img)
        assert out.shape == (3, 224, 224)
