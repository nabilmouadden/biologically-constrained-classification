import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix


def plot_training_curves(train_losses, val_losses, val_metrics, save_path=None):
    """
    Plot training curves including losses and validation metrics.
    
    Args:
        train_losses (list): Training losses for each epoch
        val_losses (list): Validation losses for each epoch
        val_metrics (list): Validation metrics (e.g., F1) for each epoch
        save_path (str, optional): Path to save the plot
    """
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot metrics
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_metrics, 'g-', label='Validation F1')
    plt.title('Validation F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Training curves saved to {save_path}")
    
    plt.close()


def visualize_constraint_matrix(model, class_names=None, save_path=None, data_loader=None, device=None):
    """
    Visualize the learned constraint matrix.
    
    Args:
        model: The trained model
        class_names (list, optional): Names of the classes
        save_path (str, optional): Path to save the visualization
        data_loader (DataLoader, optional): If provided, computes averaged constraint matrix
        device (torch.device, optional): Device to use for computation
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    # If data_loader is provided, compute average constraint matrix
    if data_loader is not None:
        constraint_matrices = []
        with torch.no_grad():
            for images, _ in data_loader:
                images = images.to(device)
                outputs = model(images, training=False)
                constraint_matrices.append(outputs['constraint_matrix'].cpu().numpy())
        
        # Average the constraint matrices
        avg_constraint_matrix = np.mean(np.concatenate(constraint_matrices, axis=0), axis=0)
    else:
        # Extract the constraint matrix from the model
        # This is a simplification - actual implementation may vary based on model structure
        # Here we create a dummy input to get the matrix
        dummy_input = torch.ones(1, 3, 224, 224).to(device)  # Adjust shape as needed
        with torch.no_grad():
            outputs = model(dummy_input, training=False)
        avg_constraint_matrix = outputs['constraint_matrix'][0].cpu().numpy()
    
    # Get prior constraint matrix
    prior_matrix = model.constraint_module.prior_constraints.cpu().numpy()
    
    # If class names not provided, use generic labels
    if class_names is None:
        class_names = [f"Class {i+1}" for i in range(avg_constraint_matrix.shape[0])]
    
    # Create figure with two heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Plot prior constraint matrix
    sns.heatmap(prior_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
                xticklabels=class_names, yticklabels=class_names, ax=axes[0], fmt='.2f')
    axes[0].set_title('Prior Constraint Matrix (C)')
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Class')
    
    # Plot learned constraint matrix
    sns.heatmap(avg_constraint_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
                xticklabels=class_names, yticklabels=class_names, ax=axes[1], fmt='.2f')
    axes[1].set_title('Learned Constraint Matrix (R)')
    axes[1].set_xlabel('Class')
    axes[1].set_ylabel('Class')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Constraint matrix visualization saved to {save_path}")
    
    plt.close()


def visualize_thresholds(model, class_names=None, save_path=None):
    """
    Visualize the learned adaptive thresholds.
    
    Args:
        model: The trained model
        class_names (list, optional): Names of the classes
        save_path (str, optional): Path to save the visualization
    """
    threshold_module = model.constraint_module.adaptive_threshold
    
    # Extract learnable parameters
    base_threshold = threshold_module.base_threshold
    alpha = threshold_module.threshold_alpha.cpu().numpy()
    beta = threshold_module.threshold_beta.cpu().numpy()
    delta = threshold_module.threshold_delta.cpu().numpy()
    
    # If class names not provided, use generic labels
    if class_names is None:
        class_names = [f"Class {i+1}" for i in range(len(alpha))]
    
    # Compute base thresholds
    base_thresholds = alpha * base_threshold
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot parameters as grouped bar chart
    x = np.arange(len(class_names))
    width = 0.2
    
    plt.bar(x - width*1.5, base_thresholds, width, label='Base Threshold (α·t_base)')
    plt.bar(x - width/2, beta, width, label='Uncertainty Weight (β)')
    plt.bar(x + width/2, delta, width, label='Probability Weight (δ)')
    plt.bar(x + width*1.5, base_thresholds + 0.5*beta + 0.5*delta, width, 
            label='Approx. Threshold for 0.5 prob & uncertainty')
    
    plt.xlabel('Class')
    plt.ylabel('Parameter Value')
    plt.title('Adaptive Threshold Parameters')
    plt.xticks(x, class_names, rotation=45, ha='right')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Threshold visualization saved to {save_path}")
    
    plt.close()


def plot_uncertainty_vs_accuracy(uncertainties, predictions, targets, class_names=None, save_path=None):
    """
    Plot relationship between model uncertainty and prediction accuracy.
    
    Args:
        uncertainties (numpy.ndarray): Model uncertainty for each prediction [N, K]
        predictions (numpy.ndarray): Model predictions [N, K]
        targets (numpy.ndarray): Ground truth labels [N, K]
        class_names (list, optional): Names of the classes
        save_path (str, optional): Path to save the visualization
    """
    num_classes = uncertainties.shape[1]
    
    # If class names not provided, use generic labels
    if class_names is None:
        class_names = [f"Class {i+1}" for i in range(num_classes)]
    
    plt.figure(figsize=(12, 8))
    
    # For each class
    for c in range(num_classes):
        class_uncertainty = uncertainties[:, c]
        class_correct = (predictions[:, c] == targets[:, c])
        
        # Create uncertainty bins
        bins = np.linspace(0, class_uncertainty.max() + 1e-6, 11)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        # Calculate accuracy in each bin
        bin_accuracies = []
        for i in range(len(bins) - 1):
            mask = (class_uncertainty >= bins[i]) & (class_uncertainty < bins[i+1])
            if np.sum(mask) > 0:
                bin_accuracies.append(np.mean(class_correct[mask]))
            else:
                bin_accuracies.append(np.nan)
        
        # Plot uncertainty vs accuracy
        plt.plot(bin_centers, bin_accuracies, 'o-', label=class_names[c])
    
    plt.xlabel('Uncertainty')
    plt.ylabel('Accuracy')
    plt.title('Relationship Between Uncertainty and Accuracy')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Uncertainty vs. accuracy plot saved to {save_path}")
    
    plt.close()
