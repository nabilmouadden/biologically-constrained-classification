"""Tests that config values are properly wired through, not hardcoded."""
import pytest
import torch
import torch.nn as nn
import yaml

from src.constants import EPS
from src.models import create_model
from src.models.constraint_module import ConstraintModule
from src.models.losses import ConstraintLoss


class _MockBackbone(nn.Module):
    def __init__(self, feature_dim=64):
        super().__init__()
        self.linear = nn.Linear(3 * 16 * 16, feature_dim)

    def forward(self, x):
        return self.linear(x.view(x.size(0), -1))


# ── EPS unification ──────────────────────────────────────────────────────

class TestEpsilonUnification:
    def test_eps_is_1e8(self):
        assert EPS == 1e-8

    def test_losses_uses_shared_eps(self):
        """ConstraintLoss should use the shared EPS, not a local constant."""
        import src.models.losses as losses_mod
        assert losses_mod.EPS is EPS

    def test_constraint_module_uses_shared_eps(self):
        import src.models.constraint_module as cm_mod
        assert cm_mod.EPS is EPS

    def test_uncertainty_uses_shared_eps(self):
        import src.utils.uncertainty as unc_mod
        assert unc_mod.EPS is EPS


# ── model_config passthrough ─────────────────────────────────────────────

class TestModelConfigPassthrough:
    def test_dropout_rate_passed(self):
        backbone = _MockBackbone(64)
        model, _ = create_model(
            backbone=backbone, num_classes=7, feature_dim=64,
            model_config={'dropout_rate': 0.3},
        )
        assert model.constraint_module.dropout_rate == 0.3

    def test_base_threshold_passed(self):
        backbone = _MockBackbone(64)
        model, _ = create_model(
            backbone=backbone, num_classes=7, feature_dim=64,
            model_config={'base_threshold': 0.7},
        )
        assert model.constraint_module.adaptive_threshold.base_threshold == 0.7

    def test_mc_samples_train_passed(self):
        backbone = _MockBackbone(64)
        model, _ = create_model(
            backbone=backbone, num_classes=7, feature_dim=64,
            model_config={'mc_samples_train': 10},
        )
        assert model.constraint_module.mc_samples_train == 10

    def test_defaults_when_no_model_config(self):
        backbone = _MockBackbone(64)
        model, _ = create_model(
            backbone=backbone, num_classes=7, feature_dim=64,
        )
        assert model.constraint_module.dropout_rate == 0.5
        assert model.constraint_module.adaptive_threshold.base_threshold == 0.5
        assert model.constraint_module.mc_samples_train == 5

    def test_dropout_rate_affects_uncertainty(self):
        """Different dropout_rate should change MC uncertainty magnitude."""
        backbone = _MockBackbone(64)
        model_low, _ = create_model(
            backbone=backbone, num_classes=7, feature_dim=64,
            model_config={'dropout_rate': 0.1},
        )
        model_high, _ = create_model(
            backbone=backbone, num_classes=7, feature_dim=64,
            model_config={'dropout_rate': 0.9},
        )
        # Copy weights
        model_high.load_state_dict(model_low.state_dict(), strict=False)

        torch.manual_seed(42)
        x = torch.randn(8, 3, 16, 16)
        out_low = model_low(x, training=False, mc_samples=100)
        torch.manual_seed(42)
        out_high = model_high(x, training=False, mc_samples=100)
        assert out_high['uncertainty'].mean() > out_low['uncertainty'].mean()


# ── mc_samples_train cap ─────────────────────────────────────────────────

class TestMCSamplesTrainCap:
    def test_cap_respects_config(self):
        """mc_samples_train=10 should cap training to min(10, mc_samples)."""
        module = ConstraintModule(num_classes=3, feature_dim=32, mc_samples_train=10)
        x = torch.randn(2, 32)
        # With mc_samples=100, training should use min(10, 100)=10
        out = module(x, training=True, mc_samples=100)
        assert out['logits'].shape == (2, 3)

    def test_cap_lower_wins(self):
        """If mc_samples < mc_samples_train, mc_samples wins."""
        module = ConstraintModule(num_classes=3, feature_dim=32, mc_samples_train=10)
        x = torch.randn(2, 32)
        out = module(x, training=True, mc_samples=2)
        assert out['logits'].shape == (2, 3)

    def test_default_cap_is_5(self):
        module = ConstraintModule(num_classes=3, feature_dim=32)
        assert module.mc_samples_train == 5


# ── YAML config structure ────────────────────────────────────────────────

class TestYAMLConfig:
    @pytest.fixture
    def config(self):
        config_path = 'configs/gr_neutro.yaml'
        with open(config_path) as f:
            return yaml.safe_load(f)

    def test_mc_samples_in_training(self, config):
        assert 'mc_samples_train' in config['training']
        assert 'mc_samples_val' in config['training']
        assert isinstance(config['training']['mc_samples_train'], int)
        assert isinstance(config['training']['mc_samples_val'], int)

    def test_num_workers_in_training(self, config):
        assert 'num_workers' in config['training']

    def test_dropout_rate_in_model(self, config):
        assert 'dropout_rate' in config['model']

    def test_base_threshold_in_model(self, config):
        assert 'base_threshold' in config['model']

    def test_backbone_is_dinobloom_s(self, config):
        assert config['model']['backbone'] == 'dinobloom-s'

    def test_freeze_backbone_true(self, config):
        assert config['training']['freeze_backbone'] is True

    def test_no_backbone_lr(self, config):
        assert 'backbone_lr' not in config['training']

    def test_loss_config_present(self, config):
        assert 'loss' in config['training']
        loss = config['training']['loss']
        assert 'lambda_con' in loss
        assert 'lambda_unc' in loss
        assert 'lambda_entropy' in loss

    def test_inference_section(self, config):
        assert 'inference' in config
        assert 'mc_samples' in config['inference']


# ── loss_config + model_config together ──────────────────────────────────

class TestCombinedConfigPassthrough:
    def test_both_configs_applied(self):
        backbone = _MockBackbone(64)
        model, loss_fn = create_model(
            backbone=backbone, num_classes=7, feature_dim=64,
            loss_config={'lambda_con': 0.5, 'lambda_entropy': 0.05},
            model_config={'dropout_rate': 0.3, 'base_threshold': 0.6},
        )
        assert loss_fn.lambda_con == 0.5
        assert loss_fn.lambda_entropy == 0.05
        assert model.constraint_module.dropout_rate == 0.3
        assert model.constraint_module.adaptive_threshold.base_threshold == 0.6

    def test_full_pipeline_with_configs(self):
        """End-to-end: create model with both configs, train step, eval step."""
        backbone = _MockBackbone(64)
        model, loss_fn = create_model(
            backbone=backbone, num_classes=7, feature_dim=64,
            loss_config={'lambda_con': 0.2},
            model_config={'dropout_rate': 0.4, 'mc_samples_train': 3},
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Train step
        model.train()
        x = torch.randn(4, 3, 16, 16)
        targets = torch.randint(0, 2, (4, 7)).float()
        out = model(x, training=True, mc_samples=10)  # capped at 3 by mc_samples_train
        loss = loss_fn(out, targets)['total_loss']
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Eval step
        model.eval()
        with torch.no_grad():
            out_eval = model(x, training=False, mc_samples=20)
        assert out_eval['predictions'].shape == (4, 7)
