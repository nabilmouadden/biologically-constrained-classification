import torch
import torch.nn as nn
import torch.nn.functional as F

from .adaptive_threshold import AdaptiveThreshold


class ConstraintModule(nn.Module):
    """
    Biologically-constrained multi-label classification module that enhances
    foundation models with learnable domain knowledge constraints.
    """
    def __init__(self, num_classes, feature_dim, prior_constraint_matrix=None, 
                 dropout_rate=0.5, base_threshold=0.5):
        """
        Initialize the constraint module.
        
        Args:
            num_classes (int): Number of abnormality classes (K)
            feature_dim (int): Dimension of input features from backbone (d)
            prior_constraint_matrix (torch.Tensor, optional): Prior knowledge matrix C ∈ R^(K×K)
                                                            where -1 is mutual exclusivity,
                                                            0 is no relationship, and
                                                            positive values indicate co-occurrence
            dropout_rate (float): Dropout rate for Monte Carlo sampling
            base_threshold (float): Base threshold for classification
        """
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.dropout_rate = dropout_rate
        
        # Initialize prior constraint matrix if not provided
        if prior_constraint_matrix is None:
            self.register_buffer('prior_constraints', torch.zeros(num_classes, num_classes))
        else:
            self.register_buffer('prior_constraints', prior_constraint_matrix)
        
        # =============================================
        # 1. Feature Projection Component
        # =============================================
        self.feature_projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.LayerNorm(feature_dim)
        )
        
        # =============================================
        # 2. Constraint Matrix Generation Component
        # =============================================
        # Q, K, V projections for multi-head attention
        self.query_proj = nn.Linear(feature_dim, num_classes * feature_dim)
        self.key_proj = nn.Linear(feature_dim, num_classes * feature_dim)
        self.value_proj = nn.Linear(feature_dim, num_classes * feature_dim)
        
        # Projection for each class to create relationships with all other classes
        # Maps from feature_dim to num_classes for each class representation
        self.constraint_proj = nn.Linear(feature_dim, num_classes)
        
        # =============================================
        # 3. Classification Component
        # =============================================
        self.classifier = nn.Linear(feature_dim, num_classes)
        
        # =============================================
        # 4. Adaptive Thresholding Component
        # =============================================
        self.adaptive_threshold = AdaptiveThreshold(
            num_classes=num_classes,
            base_threshold=base_threshold
        )
    
    def forward(self, features, training=True, mc_samples=50):
        """
        Forward pass through the constraint module.
        
        Args:
            features (torch.Tensor): Features from the backbone model, shape [B, N, D] or [B, D]
            training (bool): Whether the model is in training mode
            mc_samples (int): Number of Monte Carlo samples for uncertainty estimation
            
        Returns:
            dict: Dict containing predictions, constraint matrix, and other outputs
        """
        
        # Mean pooling over patches/tokens if needed
        if len(features.shape) == 3:  # [B, N, D]
            pooled_features = features.mean(dim=1)  # [B, D]
        else:  # Already pooled [B, D]
            pooled_features = features
            
        # 1. Feature Projection
        f = self.feature_projection(pooled_features)  # [B, D]
        
        # 2. Generate Constraint Matrix R
        R = self._generate_constraint_matrix(f)  # [B, K, K]
        
        # 3. Classification with uncertainty estimation
        if training:
            # During training, use fewer MC samples for efficiency
            train_mc_samples = min(5, mc_samples)
            logits, uncertainty = self._mc_forward(f, train_mc_samples)
        else:
            # During inference, use full number of MC samples
            logits, uncertainty = self._mc_forward(f, mc_samples)
        
        # 4. Apply adaptive thresholding
        probs = torch.sigmoid(logits)
        thresholds = self.adaptive_threshold(uncertainty, probs)
        predictions = (probs > thresholds).float()
        
        return {
            'logits': logits,
            'probs': probs,
            'predictions': predictions,
            'uncertainty': uncertainty,
            'constraint_matrix': R,
            'thresholds': thresholds,
            'prior_constraint_matrix': self.prior_constraints
        }
    
    def _generate_constraint_matrix(self, features):
        """
        Generate the learnable constraint matrix R using attention mechanism.
        
        This is the core implementation of the constraint matrix generation
        described in the paper. It uses a modified attention mechanism to
        create a K×K matrix that captures pairwise relationships between
        abnormality classes.
        
        Args:
            features (torch.Tensor): Projected features, shape [B, D]
            
        Returns:
            torch.Tensor: Constraint matrix R, shape [B, K, K]
        """
        batch_size = features.shape[0]
        
        # Project features to Q, K, V
        q = self.query_proj(features).view(batch_size, self.num_classes, self.feature_dim)
        k = self.key_proj(features).view(batch_size, self.num_classes, self.feature_dim)
        v = self.value_proj(features).view(batch_size, self.num_classes, self.feature_dim)
        
        # Compute attention scores with scaling factor
        attention_scores = torch.bmm(q, k.transpose(1, 2)) / (self.feature_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply attention to generate initial output
        attention_output = torch.bmm(attention_weights, v)  # [B, K, D]
        
        # Create K×K matrices that represent pairwise relationships between classes
        # We project each class vector (D-dim) to K-dim to create relationships with all K classes
        class_relationships = []
        
        # For each class, create its relationship with all other classes
        for i in range(self.num_classes):
            # Extract class representation: [B, D]
            class_repr = attention_output[:, i, :]
            
            # Project to relationship vector: [B, K]
            # This represents the relationship of class i with all K classes
            class_rel = self.constraint_proj(class_repr)
            
            # Add to list
            class_relationships.append(class_rel.unsqueeze(1))
        
        # Stack along dimension 1 to create [B, K, K]
        raw_R = torch.cat(class_relationships, dim=1)
        
        # Scale and clamp values to [-1, 1]
        # 1. Center values around mean
        mean_R = raw_R.mean(dim=(1, 2), keepdim=True)
        centered_R = raw_R - mean_R
        
        # 2. Scale so that max absolute deviation maps to 1
        max_abs_dev = centered_R.abs().max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0] + 1e-8
        alpha = 1.0 / max_abs_dev
        scaled_R = alpha * centered_R
        
        # 3. Clamp values to [-1, 1]
        R = torch.clamp(scaled_R, -1.0, 1.0)
        
        return R
    
    def _mc_forward(self, features, num_samples):
        """
        Perform Monte Carlo dropout for uncertainty estimation.
        
        This implements the uncertainty estimation approach from the paper,
        using Monte Carlo dropout to estimate predictive uncertainty.
        
        Args:
            features (torch.Tensor): Feature representations, shape [B, D]
            num_samples (int): Number of MC samples
            
        Returns:
            tuple: (mean_logits, uncertainty)
        """
        batch_size = features.shape[0]
        
        # Initialize tensors for MC sampling results
        mc_logits = torch.zeros(num_samples, batch_size, self.num_classes, 
                                device=features.device)
        
        # Perform MC sampling
        for i in range(num_samples):
            # Apply dropout to features
            dropped_features = F.dropout(features, p=self.dropout_rate, training=True)
            
            # Get logits from the classifier
            logits = self.classifier(dropped_features)
            mc_logits[i] = logits
        
        # Compute mean logits
        mean_logits = mc_logits.mean(dim=0)  # [B, K]
        
        # Convert logits to probabilities
        probs = torch.sigmoid(mc_logits)
        mean_probs = probs.mean(dim=0)  # [B, K]
        
        # Compute uncertainty (predictive variance)
        # Uncertainty = 1/M ∑(p_i - p_mean)²
        uncertainty = ((probs - mean_probs.unsqueeze(0)) ** 2).mean(dim=0)  # [B, K]
        
        return mean_logits, uncertainty
