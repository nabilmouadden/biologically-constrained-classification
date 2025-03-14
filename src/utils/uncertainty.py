import torch
import numpy as np
from scipy import stats


def entropy(probs):
    """
    Compute entropy of probability distributions.
    
    Args:
        probs (torch.Tensor): Probability distributions, shape [B, K]
        
    Returns:
        torch.Tensor: Entropy values, shape [B]
    """
    # Add small epsilon to avoid log(0)
    eps = 1e-8
    entropy_values = -torch.sum(probs * torch.log(probs + eps), dim=1)
    return entropy_values


def predictive_entropy(mc_probs):
    """
    Compute predictive entropy from Monte Carlo samples.
    
    Args:
        mc_probs (torch.Tensor): MC probability samples, shape [M, B, K]
        
    Returns:
        torch.Tensor: Predictive entropy, shape [B, K]
    """
    # Average probabilities
    mean_probs = torch.mean(mc_probs, dim=0)  # [B, K]
    
    # Compute entropy of the mean probabilities
    return -mean_probs * torch.log(mean_probs + 1e-8)


def mutual_information(mc_probs):
    """
    Compute mutual information (epistemic uncertainty) from Monte Carlo samples.
    
    Args:
        mc_probs (torch.Tensor): MC probability samples, shape [M, B, K]
        
    Returns:
        torch.Tensor: Mutual information, shape [B, K]
    """
    # Average probabilities
    mean_probs = torch.mean(mc_probs, dim=0)  # [B, K]
    
    # Average entropy
    sample_entropies = -mc_probs * torch.log(mc_probs + 1e-8)  # [M, B, K]
    avg_sample_entropy = torch.mean(torch.sum(sample_entropies, dim=2), dim=0)  # [B]
    
    # Entropy of average probabilities
    entropy_avg_probs = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8), dim=1)  # [B]
    
    # Mutual information = entropy of average probs - average entropy
    return entropy_avg_probs - avg_sample_entropy


def calibration_metrics(probs, labels, n_bins=10):
    """
    Compute calibration metrics: ECE (Expected Calibration Error) and MCE (Maximum Calibration Error).
    
    Args:
        probs (np.ndarray): Predicted probabilities, shape [N, K] or [N]
        labels (np.ndarray): True binary labels, shape [N, K] or [N]
        n_bins (int): Number of bins for calibration
        
    Returns:
        dict: Dictionary with calibration metrics
    """
    if len(probs.shape) == 1:
        probs = probs.reshape(-1, 1)
        labels = labels.reshape(-1, 1)
    
    n_classes = probs.shape[1]
    
    # Initialize results
    ece_values = []
    mce_values = []
    bin_confs = []
    bin_accs = []
    bin_sizes = []
    
    for c in range(n_classes):
        class_probs = probs[:, c]
        class_labels = labels[:, c]
        
        # Create bins
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(class_probs, bin_edges[1:-1])
        
        # Initialize arrays for class
        class_bin_confs = np.zeros(n_bins)
        class_bin_accs = np.zeros(n_bins)
        class_bin_sizes = np.zeros(n_bins)
        
        # Compute calibration in each bin
        for b in range(n_bins):
            bin_mask = (bin_indices == b)
            bin_size = np.sum(bin_mask)
            class_bin_sizes[b] = bin_size
            
            if bin_size > 0:
                bin_probs = class_probs[bin_mask]
                bin_labels = class_labels[bin_mask]
                
                # Bin confidence = average predicted probability
                class_bin_confs[b] = np.mean(bin_probs)
                
                # Bin accuracy = average ground truth label
                class_bin_accs[b] = np.mean(bin_labels)
        
        # Compute calibration errors
        abs_errors = np.abs(class_bin_accs - class_bin_confs)
        weighted_errors = class_bin_sizes * abs_errors / np.sum(class_bin_sizes)
        
        # Expected Calibration Error
        ece = np.sum(weighted_errors)
        ece_values.append(ece)
        
        # Maximum Calibration Error
        mce = np.max(abs_errors) if len(abs_errors) > 0 else 0
        mce_values.append(mce)
        
        # Store bins
        bin_confs.append(class_bin_confs)
        bin_accs.append(class_bin_accs)
        bin_sizes.append(class_bin_sizes)
    
    # Average across classes
    mean_ece = np.mean(ece_values)
    mean_mce = np.mean(mce_values)
    
    return {
        'ece': mean_ece,
        'mce': mean_mce,
        'ece_per_class': ece_values,
        'mce_per_class': mce_values,
        'bin_confs': bin_confs,
        'bin_accs': bin_accs,
        'bin_sizes': bin_sizes
    }


def compute_ood_scores(in_domain_probs, out_domain_probs):
    """
    Compute OOD detection metrics using probabilities.
    
    Args:
        in_domain_probs (np.ndarray): In-domain probabilities, shape [N_in, K]
        out_domain_probs (np.ndarray): Out-of-domain probabilities, shape [N_out, K]
        
    Returns:
        dict: Dictionary with OOD metrics
    """
    # Maximum softmax probability
    in_msp = np.max(in_domain_probs, axis=1)
    out_msp = np.max(out_domain_probs, axis=1)
    
    # Predictive entropy
    in_entropy = -np.sum(in_domain_probs * np.log(in_domain_probs + 1e-8), axis=1)
    out_entropy = -np.sum(out_domain_probs * np.log(out_domain_probs + 1e-8), axis=1)
    
    # Evaluate detection performance
    # AUROC
    y_true = np.concatenate([np.zeros(len(in_msp)), np.ones(len(out_msp))])
    
    # MSP score (lower score = higher probability of being OOD)
    msp_score = np.concatenate([-in_msp, -out_msp])
    msp_auroc = compute_auroc(msp_score, y_true)
    
    # Entropy score (higher entropy = higher probability of being OOD)
    entropy_score = np.concatenate([in_entropy, out_entropy])
    entropy_auroc = compute_auroc(entropy_score, y_true)
    
    return {
        'msp_auroc': msp_auroc,
        'entropy_auroc': entropy_auroc,
        'in_msp_mean': np.mean(in_msp),
        'out_msp_mean': np.mean(out_msp),
        'in_entropy_mean': np.mean(in_entropy),
        'out_entropy_mean': np.mean(out_entropy)
    }


def compute_auroc(scores, labels):
    """
    Compute AUROC score.
    
    Args:
        scores (np.ndarray): Detection scores
        labels (np.ndarray): True labels (0 for in-domain, 1 for OOD)
        
    Returns:
        float: AUROC score
    """
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(labels, scores)


def confidence_interval(data, confidence=0.95):
    """
    Compute confidence interval using t-distribution.
    
    Args:
        data (np.ndarray): Data points
        confidence (float): Confidence level
        
    Returns:
        tuple: (mean, (lower_bound, upper_bound))
    """
    n = len(data)
    mean = np.mean(data)
    se = stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean, (mean - h, mean + h)
