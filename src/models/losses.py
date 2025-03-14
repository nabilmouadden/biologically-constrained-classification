import torch
import torch.nn as nn
import torch.nn.functional as F


class ConstraintLoss(nn.Module):
    """
    Loss function for the biologically-constrained multi-label classification.
    
    This implements the complete loss function described in the paper:
    "Biologically-Constrained Multi-Label Classification with Learnable Domain Knowledge"
    
    The total loss is a weighted sum of:
    - Binary Cross-Entropy Loss
    - Constraint Loss
    - Uncertainty Loss
    - Entropy Loss
    """
    def __init__(self, lambda_con=0.1, lambda_unc=0.1, lambda_entropy=0.01, 
                 alpha=0.01, beta=0.1, uncertainty_threshold=0.2):
        """
        Initialize the constraint loss.
        
        Args:
            lambda_con (float): Weight for constraint loss
            lambda_unc (float): Weight for uncertainty loss
            lambda_entropy (float): Weight for entropy loss
            alpha (float): Weight for L1 regularization in constraint loss
            beta (float): Weight for excessive uncertainty penalty
            uncertainty_threshold (float): Threshold for acceptable uncertainty
        """
        super().__init__()
        self.lambda_con = lambda_con
        self.lambda_unc = lambda_unc
        self.lambda_entropy = lambda_entropy
        self.alpha = alpha
        self.beta = beta
        self.uncertainty_threshold = uncertainty_threshold
    
    def forward(self, outputs, targets):
        """
        Compute the total loss.
        
        Args:
            outputs (dict): Model outputs including logits, constraint matrix, etc.
            targets (torch.Tensor): Ground truth labels, shape [B, K]
            
        Returns:
            dict: Dictionary of loss components and total loss
        """
        logits = outputs['logits']
        R = outputs['constraint_matrix']
        uncertainty = outputs['uncertainty']
        C = outputs['prior_constraint_matrix']
        
        # 1. Binary Cross-Entropy Loss
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets)
        
        # 2. Constraint Loss
        constraint_loss = self._compute_constraint_loss(R, C)
        
        # 3. Uncertainty Loss
        uncertainty_loss = self._compute_uncertainty_loss(uncertainty, logits, targets)
        
        # 4. Entropy Loss
        entropy_loss = self._compute_entropy_loss(R)
        
        # Combine losses with weights
        total_loss = bce_loss + \
                     self.lambda_con * constraint_loss + \
                     self.lambda_unc * uncertainty_loss + \
                     self.lambda_entropy * entropy_loss
        
        return {
            'total_loss': total_loss,
            'bce_loss': bce_loss,
            'constraint_loss': constraint_loss,
            'uncertainty_loss': uncertainty_loss,
            'entropy_loss': entropy_loss
        }
    
    def _compute_constraint_loss(self, R, C):
        """
        Compute the constraint loss that enforces biological relationships.
        
        The constraint loss is defined as:
        L_con = ||RR^T - C||_F^2 + α||R||_1
        
        This loss aligns the learned constraint matrix R with the prior matrix C
        while encouraging sparsity through L1 regularization.
        
        Args:
            R (torch.Tensor): Learned constraint matrix, shape [B, K, K]
            C (torch.Tensor): Prior constraint matrix, shape [K, K]
            
        Returns:
            torch.Tensor: Constraint loss
        """
        # Compute R·R^T to capture direct and transitive relationships
        batch_size = R.shape[0]
        RRT = torch.bmm(R, R.transpose(1, 2))  # [B, K, K]
        
        # Expand C to match batch dimension for broadcasting
        C_expanded = C.unsqueeze(0).expand(batch_size, -1, -1)  # [B, K, K]
        
        # Compute SQUARED Frobenius norm between RRT and C
        # ||RRT - C||_F^2
        frobenius_term = torch.pow(torch.norm(RRT - C_expanded, p='fro', dim=(1, 2)), 2).mean()
        
        # L1 regularization to encourage sparsity in R
        l1_term = self.alpha * torch.norm(R, p=1, dim=(1, 2)).mean()
        
        return frobenius_term + l1_term
    
    def _compute_uncertainty_loss(self, uncertainty, logits, targets):
        """
        Compute the uncertainty loss to ensure confident predictions.
        
        The uncertainty loss has two components:
        1. KL divergence between predicted and true distributions
        2. Hinge loss to penalize excessive uncertainty
        
        Args:
            uncertainty (torch.Tensor): Predictive uncertainty, shape [B, K]
            logits (torch.Tensor): Predicted logits, shape [B, K]
            targets (torch.Tensor): Ground truth labels, shape [B, K]
            
        Returns:
            torch.Tensor: Uncertainty loss
        """
        # Get probabilities from logits
        predicted_probs = torch.sigmoid(logits)
        
        # Compute KL divergence
        eps = 1e-10
        kl_pos = targets * torch.log((targets + eps) / (predicted_probs + eps))
        kl_neg = (1 - targets) * torch.log((1 - targets + eps) / (1 - predicted_probs + eps))
        kl_div = (kl_pos + kl_neg).mean()
        
        # Hinge loss term to penalize excessive uncertainty
        # Penalizes uncertainty that exceeds the threshold
        hinge_term = self.beta * F.relu(uncertainty - self.uncertainty_threshold).mean()
        
        return kl_div + hinge_term
    
    def _compute_entropy_loss(self, R):
        """
        Compute entropy loss to prevent constraints from becoming too deterministic.
        
        This entropy regularization prevents the model from enforcing excessively
        rigid constraints, allowing flexibility in relationship learning.
        
        Args:
            R (torch.Tensor): Constraint matrix, shape [B, K, K]
            
        Returns:
            torch.Tensor: Entropy loss
        """
        # Normalize R to [0, 1] for entropy calculation
        R_normalized = (R + 1) / 2
        
        # Calculate binary entropy for each element: -[p*log(p) + (1-p)*log(1-p)]
        element_entropy = -(R_normalized * torch.log(R_normalized + 1e-10) + 
                           (1 - R_normalized) * torch.log(1 - R_normalized + 1e-10))
        
        # Sum entropy over matrix dimensions (K,K) and average over batch
        # Higher value means more uncertainty (higher entropy)
        batch_entropy = element_entropy.sum(dim=(1, 2)).mean()
        
        # Return negative entropy as the loss (to maximize entropy)
        # We want to minimize the loss, so negative entropy maximizes entropy
        return -batch_entropy