from .constraint_module import ConstraintModule
from .adaptive_threshold import AdaptiveThreshold
from .losses import ConstraintLoss
from .constraint_priors import get_constraint_matrix, save_constraint_matrix


def create_model(backbone, num_classes, feature_dim, prior_constraint_matrix=None):
    """
    Create a complete biologically-constrained classification model.
    
    Args:
        backbone (nn.Module): Foundation model backbone
        num_classes (int): Number of abnormality classes
        feature_dim (int): Dimension of backbone features
        prior_constraint_matrix (torch.Tensor, optional): Prior constraint matrix
        
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
    
    # Create constraint module
    constraint_module = ConstraintModule(
        num_classes=num_classes,
        feature_dim=feature_dim,
        prior_constraint_matrix=prior_constraint_matrix
    )
    
    # Create full model
    model = BiologicallyConstrainedModel(backbone, constraint_module)
    
    # Create loss function
    loss_fn = ConstraintLoss()
    
    return model, loss_fn


__all__ = ['ConstraintModule', 'AdaptiveThreshold', 'ConstraintLoss', 
           'create_model', 'get_constraint_matrix', 'save_constraint_matrix']
