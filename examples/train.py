import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from src.model.constraint_priors import get_constraint_matrix
# Import the model components
from src.model import create_model
from src.data import get_dataset
from src.utils.visualization import plot_training_curves, visualize_constraint_matrix


def parse_args():
    parser = argparse.ArgumentParser(description='Train a biologically-constrained classification model')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, default='./checkpoints', help='Directory to save outputs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()


def train_epoch(model, loader, optimizer, loss_fn, device):
    """Train for one epoch"""
    model.train()
    epoch_loss = 0
    loss_components = {
        'bce_loss': 0,
        'constraint_loss': 0,
        'uncertainty_loss': 0,
        'entropy_loss': 0
    }
    
    progress_bar = tqdm(loader, desc="Training")
    for batch_idx, (images, targets) in enumerate(progress_bar):
        images, targets = images.to(device), targets.to(device)
        
        # Forward pass
        outputs = model(images, training=True, mc_samples=5)
        
        # Compute loss
        loss_dict = loss_fn(outputs, targets)
        total_loss = loss_dict['total_loss']
        
        # Backward pass and optimization
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Accumulate metrics
        epoch_loss += total_loss.item()
        for k in loss_components:
            if k in loss_dict:
                loss_components[k] += loss_dict[k].item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{total_loss.item():.4f}"
        })
    
    # Average the losses
    num_batches = len(loader)
    epoch_loss /= num_batches
    for k in loss_components:
        loss_components[k] /= num_batches
    
    return epoch_loss, loss_components


def validate(model, loader, loss_fn, device):
    """Validate the model"""
    model.eval()
    val_loss = 0
    loss_components = {
        'bce_loss': 0,
        'constraint_loss': 0,
        'uncertainty_loss': 0,
        'entropy_loss': 0
    }
    all_preds = []
    all_probs = []
    all_targets = []
    all_uncertainties = []
    
    progress_bar = tqdm(loader, desc="Validating")
    with torch.no_grad():
        for images, targets in progress_bar:
            images, targets = images.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(images, training=False, mc_samples=50)
            
            # Compute loss
            loss_dict = loss_fn(outputs, targets)
            val_loss += loss_dict['total_loss'].item()
            
            # Accumulate loss components
            for k in loss_components:
                if k in loss_dict:
                    loss_components[k] += loss_dict[k].item()
            
            # Store predictions, probabilities, targets, and uncertainties
            all_preds.append(outputs['predictions'].cpu())
            all_probs.append(outputs['probs'].cpu())
            all_targets.append(targets.cpu())
            all_uncertainties.append(outputs['uncertainty'].cpu())
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss_dict['total_loss'].item():.4f}"
            })
    
    # Concatenate all tensors
    all_preds = torch.cat(all_preds, dim=0)
    all_probs = torch.cat(all_probs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_uncertainties = torch.cat(all_uncertainties, dim=0)
    
    # Compute metrics
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
    
    # Convert to numpy for scikit-learn metrics
    preds_np = all_preds.numpy()
    targets_np = all_targets.numpy()
    probs_np = all_probs.numpy()
    
    # Overall metrics
    accuracy = accuracy_score(targets_np.flatten(), preds_np.flatten())
    
    # F1 scores (macro and weighted)
    f1_macro = f1_score(targets_np, preds_np, average='macro')
    f1_weighted = f1_score(targets_np, preds_np, average='weighted')
    
    # Precision and recall
    precision = precision_score(targets_np, preds_np, average='weighted', zero_division=0)
    recall = recall_score(targets_np, preds_np, average='weighted', zero_division=0)
    
    # AUC-ROC (try-except in case of single class)
    try:
        auc = roc_auc_score(targets_np, probs_np, average='weighted')
    except Exception as e:
        print(f"Warning: Could not compute AUC-ROC: {e}")
        auc = 0.0
    
    # Average the losses
    num_batches = len(loader)
    val_loss /= num_batches
    for k in loss_components:
        loss_components[k] /= num_batches
    
    # Calculate per-class metrics
    num_classes = all_targets.shape[1]
    class_metrics = []
    
    for c in range(num_classes):
        class_preds = preds_np[:, c]
        class_targets = targets_np[:, c]
        
        class_accuracy = accuracy_score(class_targets, class_preds)
        
        try:
            class_f1 = f1_score(class_targets, class_preds)
        except:
            class_f1 = 0.0
            
        try:
            class_auc = roc_auc_score(class_targets, probs_np[:, c])
        except:
            class_auc = 0.0
        
        class_metrics.append({
            'accuracy': class_accuracy,
            'f1': class_f1,
            'auc': class_auc,
        })
    
    metrics = {
        'loss': val_loss,
        'loss_components': loss_components,
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision': precision,
        'recall': recall,
        'auc': auc,
        'class_metrics': class_metrics,
        'uncertainties': all_uncertainties.mean(dim=0).numpy().tolist()
    }
    
    return metrics


def main():
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Save the configuration
    with open(os.path.join(args.output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load datasets
    train_dataset = get_dataset(config['dataset'], split='train')
    val_dataset = get_dataset(config['dataset'], split='val')
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training'].get('num_workers', 4)
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training'].get('num_workers', 4)
    )
    
    # Load prior constraint matrix if available
    print(f"Generating constraint matrix for {config['dataset']['name']}...")
    prior_constraint_matrix = get_constraint_matrix(config['dataset']['name'])
    
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
    
    # Freeze/unfreeze backbone layers based on configuration
    freeze_backbone = config['training'].get('freeze_backbone', True)
    
    if freeze_backbone:
        print("Freezing backbone parameters")
        for param in backbone.parameters():
            param.requires_grad = False
    else:
        # Selective unfreezing based on configuration
        unfreeze_layers = config['training'].get('unfreeze_layers', [])
        
        if unfreeze_layers:
            # First freeze all parameters
            for param in backbone.parameters():
                param.requires_grad = False
                
            print(f"Selectively unfreezing backbone layers: {unfreeze_layers}")
            
            # Unfreeze specific transformer blocks
            if 'blocks' in unfreeze_layers:
                block_indices = config['training'].get('unfreeze_block_indices', [])
                if block_indices:
                    for idx in block_indices:
                        if hasattr(backbone, 'blocks') and idx < len(backbone.blocks):
                            print(f"Unfreezing transformer block {idx}")
                            for param in backbone.blocks[idx].parameters():
                                param.requires_grad = True
                else:
                    # If no specific blocks are specified, unfreeze the last n blocks
                    num_last_blocks = config['training'].get('unfreeze_last_n_blocks', 2)
                    if hasattr(backbone, 'blocks'):
                        for idx in range(max(0, len(backbone.blocks) - num_last_blocks), len(backbone.blocks)):
                            print(f"Unfreezing transformer block {idx}")
                            for param in backbone.blocks[idx].parameters():
                                param.requires_grad = True
            
            # Unfreeze patch embedding
            if 'patch_embed' in unfreeze_layers and hasattr(backbone, 'patch_embed'):
                print("Unfreezing patch embedding")
                for param in backbone.patch_embed.parameters():
                    param.requires_grad = True
            
            # Unfreeze norm layer (pre-head normalization)
            if 'norm' in unfreeze_layers and hasattr(backbone, 'norm'):
                print("Unfreezing normalization layer")
                for param in backbone.norm.parameters():
                    param.requires_grad = True
        else:
            print("Unfreezing all backbone parameters")
            for param in backbone.parameters():
                param.requires_grad = True
    
    # Create model and loss function
    model, loss_fn = create_model(
        backbone=backbone,
        num_classes=config['model']['num_classes'],
        feature_dim=feature_dim,
        prior_constraint_matrix=prior_constraint_matrix
    )
    model = model.to(device)
    
    # Set up optimizer with different learning rates for components
    optimizer = optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': config['training']['backbone_lr']},
        {'params': model.constraint_module.feature_projection.parameters(), 'lr': config['training']['constraint_lr']},
        {'params': model.constraint_module.query_proj.parameters(), 'lr': config['training']['constraint_lr']},
        {'params': model.constraint_module.key_proj.parameters(), 'lr': config['training']['constraint_lr']},
        {'params': model.constraint_module.value_proj.parameters(), 'lr': config['training']['constraint_lr']},
        {'params': model.constraint_module.constraint_proj.parameters(), 'lr': config['training']['constraint_lr']},
        {'params': model.constraint_module.classifier.parameters(), 'lr': config['training']['classifier_lr']},
        {'params': model.constraint_module.adaptive_threshold.parameters(), 'lr': config['training']['threshold_lr']}
    ], weight_decay=config['training'].get('weight_decay', 0.01))
    
    # Set up learning rate scheduler if specified
    if 'scheduler' in config['training']:
        if config['training']['scheduler']['type'] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config['training']['epochs'],
                eta_min=config['training']['scheduler'].get('min_lr', 1e-6)
            )
        else:
            scheduler = None
    else:
        scheduler = None
    
    # Training loop
    best_val_f1 = 0.0
    early_stopping_patience = config['training'].get('early_stopping_patience', 10)
    early_stopping_counter = 0
    train_losses = []
    val_losses = []
    val_f1s = []
    
    for epoch in range(config['training']['epochs']):
        print(f"Epoch {epoch+1}/{config['training']['epochs']}")
        
        # Train
        train_loss, train_loss_components = train_epoch(model, train_loader, optimizer, loss_fn, device)
        train_losses.append(train_loss)
        
        # Validate
        val_metrics = validate(model, val_loader, loss_fn, device)
        val_loss = val_metrics['loss']
        val_f1 = val_metrics['f1_weighted']
        val_losses.append(val_loss)
        val_f1s.append(val_f1)
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Val F1: {val_f1:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
        
        # Print loss components
        print("Loss components:")
        for k, v in train_loss_components.items():
            print(f"  {k}: {v:.4f}")
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            early_stopping_counter = 0
            print(f"New best model with F1: {val_f1:.4f}")
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'config': config
            }, os.path.join(args.output_dir, 'best_model.pth'))
        else:
            early_stopping_counter += 1
            print(f"No improvement for {early_stopping_counter} epochs. Best F1: {best_val_f1:.4f}")
        
        # Early stopping
        if early_stopping_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
        
        # Save checkpoint
        if (epoch + 1) % config['training'].get('checkpoint_interval', 5) == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'config': config
            }, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pth'))
        
        # Visualize constraint matrix
        if (epoch + 1) % config['training'].get('visualization_interval', 10) == 0:
            visualize_constraint_matrix(
                model, 
                class_names=config['dataset']['class_names'],
                save_path=os.path.join(args.output_dir, f'constraint_matrix_epoch_{epoch+1}.png')
            )
    
    # Plot training curves
    plot_training_curves(
        train_losses, val_losses, val_f1s,
        save_path=os.path.join(args.output_dir, 'training_curves.png')
    )
    
    print(f"Best validation F1: {best_val_f1:.4f}")
    print(f"Training complete. Models saved to {args.output_dir}")