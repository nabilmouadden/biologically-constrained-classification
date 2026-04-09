from .constraint_module import ConstraintModule
from .adaptive_threshold import AdaptiveThreshold
from .losses import ConstraintLoss
from .constraint_priors import get_constraint_matrix, save_constraint_matrix


def create_model(backbone, num_classes, feature_dim, prior_constraint_matrix=None,
                 loss_config=None, model_config=None):
    """
    Create a complete biologically-constrained classification model.

    Args:
        backbone (nn.Module): Foundation model backbone
        num_classes (int): Number of abnormality classes
        feature_dim (int): Dimension of backbone features
        prior_constraint_matrix (torch.Tensor, optional): Prior constraint matrix
        loss_config (dict, optional): Loss function hyperparameters from config YAML
        model_config (dict, optional): Model hyperparameters (dropout_rate, base_threshold, etc.)

    Returns:
        tuple: (model, loss_function)
    """
    from torch import nn

    class BiologicallyConstrainedModel(nn.Module):
        def __init__(self, backbone, constraint_module):
            super().__init__()
            self.backbone = backbone
            self.constraint_module = constraint_module

        def forward(self, x, training=True, mc_samples=50):
            features = self.backbone(x)
            outputs = self.constraint_module(features, training, mc_samples)
            return outputs

    # Extract model config values with sensible defaults
    _mc = model_config or {}
    dropout_rate = _mc.get('dropout_rate', 0.5)
    base_threshold = _mc.get('base_threshold', 0.5)
    mc_samples_train = _mc.get('mc_samples_train', 5)

    # Create constraint module
    constraint_module = ConstraintModule(
        num_classes=num_classes,
        feature_dim=feature_dim,
        prior_constraint_matrix=prior_constraint_matrix,
        dropout_rate=dropout_rate,
        base_threshold=base_threshold,
        mc_samples_train=mc_samples_train,
    )
    
    # Create full model
    model = BiologicallyConstrainedModel(backbone, constraint_module)
    
    # Create loss function with config if provided
    if loss_config:
        loss_fn = ConstraintLoss(**loss_config)
    else:
        loss_fn = ConstraintLoss()
    
    return model, loss_fn


__all__ = ['ConstraintModule', 'AdaptiveThreshold', 'ConstraintLoss', 
           'create_model', 'get_constraint_matrix', 'save_constraint_matrix']
