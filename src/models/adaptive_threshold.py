import torch
import torch.nn as nn


class AdaptiveThreshold(nn.Module):
    """
    Adaptive thresholding mechanism for multi-label classification.
    
    This module implements the adaptive thresholding component described in the paper:
    "Biologically-Constrained Multi-Label Classification with Learnable Domain Knowledge"
    
    The thresholds are dynamically adjusted based on prediction confidence,
    model uncertainty, and class priors.
    """
    def __init__(self, num_classes, base_threshold=0.5):
        """
        Initialize the adaptive thresholding module.
        
        Args:
            num_classes (int): Number of abnormality classes
            base_threshold (float): Base threshold value (default: 0.5)
        """
        super().__init__()
        self.num_classes = num_classes
        self.base_threshold = base_threshold
        
        # Learnable parameters for adaptive thresholding
        # α_i: weight for base threshold contribution
        self.threshold_alpha = nn.Parameter(torch.ones(num_classes))
        
        # β_i: weight for uncertainty contribution
        self.threshold_beta = nn.Parameter(torch.zeros(num_classes))
        
        # δ_i: weight for class frequency contribution
        self.threshold_delta = nn.Parameter(torch.zeros(num_classes))
    
    def forward(self, uncertainty, probabilities):
        """
        Compute adaptive thresholds for each class.
        
        The threshold for each class is calculated as:
        t_i = α_i · t_base + β_i · U_i + δ_i · p_i
        
        Where:
        - t_base is the base threshold
        - U_i is the uncertainty for class i
        - p_i is the predicted probability for class i
        - α_i, β_i, δ_i are learnable parameters
        
        Args:
            uncertainty (torch.Tensor): Predictive uncertainty, shape [B, K]
            probabilities (torch.Tensor): Predicted probabilities, shape [B, K]
            
        Returns:
            torch.Tensor: Adaptive thresholds, shape [B, K]
        """
        # Base threshold component
        base_component = self.threshold_alpha * self.base_threshold
        
        # Uncertainty component (higher uncertainty → higher threshold)
        uncertainty_component = self.threshold_beta * uncertainty
        
        # Probability component (class frequency adjustment)
        probability_component = self.threshold_delta * probabilities
        
        # Combine components
        thresholds = base_component.unsqueeze(0) + uncertainty_component + probability_component
        
        # Ensure thresholds are between 0 and 1
        thresholds = torch.clamp(thresholds, 0.0, 1.0)
        
        return thresholds
