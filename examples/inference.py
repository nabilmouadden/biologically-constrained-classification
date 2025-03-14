import os
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score, roc_auc_score

# Import the model components
from src.model import create_model
from src.data import get_dataset
from src.utils.visualization import visualize_constraint_matrix, visualize_thresholds, plot_uncertainty_vs_accuracy
from src.utils.uncertainty import calibration_metrics


def parse_args():
    parser = argparse.ArgumentParser(description='Run inference with a trained model')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save outputs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference')
    parser.add_argument('--mc_samples', type=int, default=50, help='Number of Monte Carlo samples')
    return parser.parse_args()


def load_model(config, checkpoint_path, device):
    """
    Load a trained model from checkpoint.
    
    Args:
        config (dict): Model configuration
        checkpoint_path (str): Path to checkpoint file
        device (torch.device): Device to load the model onto
        
    Returns:
        model: Loaded model
    """
    # Load backbone model
    if config['model']['backbone'] == 'dinobloom-s':
        from torchvision.models import dinov2_vits14
        backbone = dinov2_vits14(pretrained=True)
        feature_dim = 384
    elif config['model']['backbone'] == 'dinobloom-b':
        from torchvision.models import dinov2_vitb14
        backbone = dinov2_vitb14(pretrained=True)
        feature_dim = 768
    elif config['model']['backbone'] == 'dinobloom-l':
        from torchvision.models import dinov2_vitl14
        backbone = dinov2_vitl14(pretrained=True)
        feature_dim = 1024
    else:
        raise ValueError(f"Unsupported backbone: {config['model']['backbone']}")
    
    # Load prior constraint matrix if available
    if 'prior_constraint_matrix' in config:
        prior_matrix_path = config['prior_constraint_matrix']
        if os.path.exists(prior_matrix_path):
            print(f"Loading prior constraint matrix from {prior_matrix_path}")
            prior_constraint_matrix = torch.load(prior_matrix_path)
        else:
            print(f"Prior constraint matrix path {prior_matrix_path} not found.")
            print(f"Generating constraint matrix for {config['dataset']['name']}...")
            from src.model.constraint_priors import get_constraint_matrix
            prior_constraint_matrix = get_constraint_matrix(config['dataset']['name'])
    else:
        # Generate from dataset name if not provided
        print(f"No prior constraint matrix specified.")
        print(f"Generating constraint matrix for {config['dataset']['name']}...")
        from src.model.constraint_priors import get_constraint_matrix
        prior_constraint_matrix = get_constraint_matrix(config['dataset']['name'])
    
    # Create model
    model, _ = create_model(
        backbone=backbone,
        num_classes=config['model']['num_classes'],
        feature_dim=feature_dim,
        prior_constraint_matrix=prior_constraint_matrix
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model


def run_inference(model, data_loader, device, mc_samples=50):
    """
    Run inference on a dataset.
    
    Args:
        model: The model to use for inference
        data_loader: DataLoader for the dataset
        device: Device to run inference on
        mc_samples: Number of Monte Carlo samples
        
    Returns:
        dict: Dictionary with inference results
    """
    all_probs = []
    all_preds = []
    all_targets = []
    all_uncertainties = []
    all_constraint_matrices = []
    all_thresholds = []
    
    progress_bar = tqdm(data_loader, desc="Running inference")
    with torch.no_grad():
        for images, targets in progress_bar:
            images, targets = images.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(images, training=False, mc_samples=mc_samples)
            
            # Collect outputs
            all_probs.append(outputs['probs'].cpu().numpy())
            all_preds.append(outputs['predictions'].cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_uncertainties.append(outputs['uncertainty'].cpu().numpy())
            all_constraint_matrices.append(outputs['constraint_matrix'].cpu().numpy())
            all_thresholds.append(outputs['thresholds'].cpu().numpy())
    
    # Concatenate all outputs
    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_uncertainties = np.concatenate(all_uncertainties, axis=0)
    all_constraint_matrices = np.concatenate(all_constraint_matrices, axis=0)
    all_thresholds = np.concatenate(all_thresholds, axis=0)
    
    return {
        'probs': all_probs,
        'preds': all_preds,
        'targets': all_targets,
        'uncertainties': all_uncertainties,
        'constraint_matrices': all_constraint_matrices,
        'thresholds': all_thresholds
    }

def analyze_results(results, class_names, output_dir):
    """
    Analyze inference results and save visualizations.
    
    Args:
        results (dict): Dictionary with inference results
        class_names (list): List of class names
        output_dir (str): Directory to save outputs
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract results
    probs = results['probs']
    preds = results['preds']
    targets = results['targets']
    uncertainties = results['uncertainties']
    
    # Overall metrics
    accuracy = accuracy_score(targets.flatten(), preds.flatten())
    f1_macro = f1_score(targets, preds, average='macro')
    f1_weighted = f1_score(targets, preds, average='weighted')
    precision = precision_score(targets, preds, average='weighted', zero_division=0)
    recall = recall_score(targets, preds, average='weighted', zero_division=0)
    
    try:
        auc = roc_auc_score(targets, probs, average='weighted')
    except:
        auc = np.nan
    
    print(f"Overall Metrics:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1 Score (Macro): {f1_macro:.4f}")
    print(f"  F1 Score (Weighted): {f1_weighted:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  AUC-ROC: {auc:.4f}")
    
    # Per-class metrics
    class_metrics = {}
    for i, class_name in enumerate(class_names):
        class_targets = targets[:, i]
        class_preds = preds[:, i]
        class_probs = probs[:, i]
        
        class_accuracy = accuracy_score(class_targets, class_preds)
        
        try:
            class_f1 = f1_score(class_targets, class_preds)
        except:
            class_f1 = np.nan
            
        try:
            class_auc = roc_auc_score(class_targets, class_probs)
        except:
            class_auc = np.nan
        
        class_metrics[class_name] = {
            'accuracy': class_accuracy,
            'f1': class_f1,
            'auc': class_auc,
            'uncertainty': np.mean(uncertainties[:, i])
        }
    
    # Print per-class metrics
    print("\nPer-Class Metrics:")
    for class_name, metrics in class_metrics.items():
        print(f"  {class_name}:")
        print(f"    Accuracy: {metrics['accuracy']:.4f}")
        print(f"    F1 Score: {metrics['f1']:.4f}")
        print(f"    AUC-ROC: {metrics['auc']:.4f}")
        print(f"    Mean Uncertainty: {metrics['uncertainty']:.4f}")
    
    # Save per-class metrics to CSV
    df_metrics = pd.DataFrame.from_dict(class_metrics, orient='index')
    df_metrics.to_csv(os.path.join(output_dir, 'class_metrics.csv'))
    
    # Calibration analysis
    calibration_results = calibration_metrics(probs, targets)
    print(f"\nCalibration Metrics:")
    print(f"  Expected Calibration Error (ECE): {calibration_results['ece']:.4f}")
    print(f"  Maximum Calibration Error (MCE): {calibration_results['mce']:.4f}")
    
    # Plot calibration curve
    plt.figure(figsize=(10, 6))
    for i, class_name in enumerate(class_names):
        bin_confs = calibration_results['bin_confs'][i]
        bin_accs = calibration_results['bin_accs'][i]
        plt.plot(bin_confs, bin_accs, 'o-', label=class_name)
    
    # Add diagonal perfect calibration line
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title('Calibration Curve')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'calibration_curve.png'), bbox_inches='tight', dpi=300)
    
    # Plot relationship between uncertainty and accuracy
    plot_uncertainty_vs_accuracy(
        uncertainties, preds, targets, class_names,
        save_path=os.path.join(output_dir, 'uncertainty_vs_accuracy.png')
    )
    
    # Save classification report
    report = classification_report(targets, preds, target_names=class_names, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(os.path.join(output_dir, 'classification_report.csv'))
    
    # Analyze misclassifications
    # For each class, find samples with high uncertainty that were misclassified
    misclassified_indices = {}
    for i, class_name in enumerate(class_names):
        # Find indices where prediction != target
        errors = (preds[:, i] != targets[:, i])
        error_indices = np.where(errors)[0]
        
        if len(error_indices) > 0:
            # Sort by uncertainty
            error_uncertainties = uncertainties[error_indices, i]
            sorted_indices = error_indices[np.argsort(-error_uncertainties)]  # Sort descending
            
            # Store top 10 or less
            misclassified_indices[class_name] = sorted_indices[:min(10, len(sorted_indices))]
    
    # Save information about high-uncertainty misclassifications
    with open(os.path.join(output_dir, 'high_uncertainty_errors.txt'), 'w') as f:
        for class_name, indices in misclassified_indices.items():
            f.write(f"{class_name}:\n")
            for idx in indices:
                true_val = targets[idx, class_names.index(class_name)]
                pred_val = preds[idx, class_names.index(class_name)]
                uncertainty = uncertainties[idx, class_names.index(class_name)]
                f.write(f"  Sample {idx}: True={true_val}, Pred={pred_val}, Uncertainty={uncertainty:.4f}\n")
            f.write("\n")


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Save config to output directory
    with open(os.path.join(args.output_dir, 'inference_config.yaml'), 'w') as f:
        yaml.dump({**config, 'checkpoint': args.checkpoint}, f)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = load_model(config, args.checkpoint, device)
    
    # Load dataset
    test_dataset = get_dataset(config['dataset'], split='test')
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config.get('num_workers', 4)
    )
    
    # Run inference
    print(f"Running inference with {args.mc_samples} Monte Carlo samples...")
    results = run_inference(model, test_loader, device, mc_samples=args.mc_samples)
    
    # Save raw results
    np.savez(
        os.path.join(args.output_dir, 'inference_results.npz'),
        probs=results['probs'],
        preds=results['preds'],
        targets=results['targets'],
        uncertainties=results['uncertainties']
    )
    
    # Analyze results
    class_names = config['dataset'].get('class_names', None)
    if class_names is None:
        if hasattr(test_dataset, 'class_names'):
            class_names = test_dataset.class_names
        else:
            class_names = [f"Class {i+1}" for i in range(results['probs'].shape[1])]
    
    analyze_results(results, class_names, args.output_dir)
    
    # Visualize model components
    print("Generating visualizations...")
    
    # Constraint matrix
    visualize_constraint_matrix(
        model, class_names,
        save_path=os.path.join(args.output_dir, 'constraint_matrix.png')
    )
    
    # Thresholds
    visualize_thresholds(
        model, class_names,
        save_path=os.path.join(args.output_dir, 'adaptive_thresholds.png')
    )
    
    print(f"Inference complete. Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()