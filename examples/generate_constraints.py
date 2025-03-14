import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from src.model.constraint_priors import save_constraint_matrix, get_constraint_matrix


def parse_args():
    parser = argparse.ArgumentParser(description='Generate constraint matrices for datasets')
    parser.add_argument('--dataset', type=str, required=True, 
                        choices=['gr_neutro', 'aml_matek', 'bmc', 'all'],
                        help='Dataset name or "all" for all datasets')
    parser.add_argument('--output_dir', type=str, default='./configs',
                        help='Directory to save constraint matrices')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize the constraint matrices')
    return parser.parse_args()


def visualize_constraint_matrix(matrix, class_names, dataset_name, output_path):
    """
    Visualize a constraint matrix as a heatmap.
    
    Args:
        matrix (numpy.ndarray): Constraint matrix
        class_names (list): List of class names
        dataset_name (str): Name of the dataset
        output_path (str): Path to save the visualization
    """
    plt.figure(figsize=(12, 10))
    
    # Create heatmap
    sns.heatmap(matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1,
                xticklabels=class_names, yticklabels=class_names, fmt='.2f')
    
    plt.title(f'Constraint Matrix for {dataset_name}')
    plt.tight_layout()
    
    # Save the visualization
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Visualization saved to {output_path}")
    plt.close()


def get_class_names(dataset_name):
    """
    Get class names for a dataset.
    
    Args:
        dataset_name (str): Name of the dataset
        
    Returns:
        list: List of class names
    """
    if dataset_name == 'gr_neutro':
        return [
            'Normal', 'Chromatin', 'Dohle', 'Hypergran',
            'Hyperseg', 'Hypogran', 'Hyposeg'
        ]
    elif dataset_name == 'aml_matek':
        return [
            "Myeloblast", "Promyelocyte", "Myelocyte", "Metamyelocyte",
            "Band Neutrophil", "Segmented Neutrophil", "Eosinophil", "Basophil",
            "Monocyte", "Lymphocyte", "Plasma Cell", "Erythroblast",
            "RBC/Platelet", "Rare/Atypical", "Artifact"
        ]
    elif dataset_name == 'bmc':
        return [
            "Myeloblast", "Promyelocyte", "Myelocyte", "Metamyelocyte",
            "Band Neutrophil", "Segmented Neutrophil", "Eosinophil", "Basophil",
            "Monocyte", "Lymphocyte", "Plasma Cell", "Erythroblast",
            "Megakaryocyte", "Pro-Erythroblast", "Baso-Erythroblast",
            "Poly-Erythroblast", "Ortho-Erythroblast", "RBC",
            "Artifact", "Smudge", "Other"
        ]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine which datasets to process
    if args.dataset == 'all':
        datasets = ['gr_neutro', 'aml_matek', 'bmc']
    else:
        datasets = [args.dataset]
    
    # Process each dataset
    for dataset_name in datasets:
        # Output path for constraint matrix
        output_path = os.path.join(args.output_dir, f"{dataset_name}_constraints.pt")
        
        # Generate and save constraint matrix
        save_constraint_matrix(dataset_name, output_path)
        
        # Visualize if requested
        if args.visualize:
            # Get constraint matrix and class names
            constraint_matrix = get_constraint_matrix(dataset_name).numpy()
            class_names = get_class_names(dataset_name)
            
            # Output path for visualization
            viz_output_path = os.path.join(args.output_dir, f"{dataset_name}_constraints.png")
            
            # Create visualization
            visualize_constraint_matrix(constraint_matrix, class_names, dataset_name, viz_output_path)


if __name__ == '__main__':
    main()
