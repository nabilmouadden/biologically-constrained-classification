import torch
import numpy as np


def get_gr_neutro_constraints():
    """
    Get the constraint matrix for the GR-Neutro dataset.
    
    The constraint matrix encodes the biological relationships between 
    different neutrophil abnormalities as described in the paper.
    
    Returns:
        torch.Tensor: Constraint matrix C ∈ R^(7×7)
    """
    # Define class names for reference
    class_names = [
        'Normal',        # 0
        'Chromatin',     # 1
        'Dohle',         # 2
        'Hypergran',     # 3
        'Hyperseg',      # 4
        'Hypogran',      # 5
        'Hyposeg'        # 6
    ]
    
    # Initialize with zeros (no relationship)
    C = np.zeros((7, 7))
    
    # Mutual exclusivity constraints (-1): pairs of abnormalities that cannot co-exist
    # Normal cells cannot have any abnormality
    C[0, 1:] = -1
    C[1:, 0] = -1
    
    # Hypergranulation and Hypogranulation are mutually exclusive
    C[3, 5] = C[5, 3] = -1
    
    # Hypersegmentation and Hyposegmentation are mutually exclusive
    C[4, 6] = C[6, 4] = -1
    
    # Co-occurrence constraints (positive values):
    # These represent increased probabilities of abnormalities occurring together
    
    # Chromatin condensation relationships
    C[1, 2] = C[2, 1] = 0.3  # Chromatin and Dohle
    C[1, 3] = C[3, 1] = 0.4  # Chromatin and Hypergranulation
    C[1, 4] = C[4, 1] = 0.3  # Chromatin and Hypersegmentation
    C[1, 5] = C[5, 1] = 0.2  # Chromatin and Hypogranulation
    C[1, 6] = C[6, 1] = 0.3  # Chromatin and Hyposegmentation
    
    # Dohle bodies relationships
    C[2, 3] = C[3, 2] = 0.3  # Dohle and Hypergranulation
    C[2, 4] = C[4, 2] = 0.2  # Dohle and Hypersegmentation
    C[2, 5] = C[5, 2] = 0.2  # Dohle and Hypogranulation
    C[2, 6] = C[6, 2] = 0.2  # Dohle and Hyposegmentation
    
    # Other relationships
    C[3, 4] = C[4, 3] = 0.3  # Hypergranulation and Hypersegmentation
    C[3, 6] = C[6, 3] = 0.2  # Hypergranulation and Hyposegmentation
    C[4, 5] = C[5, 4] = 0.2  # Hypersegmentation and Hypogranulation
    C[5, 6] = C[6, 5] = 0.2  # Hypogranulation and Hyposegmentation
    
    return torch.tensor(C, dtype=torch.float32)


def get_aml_matek_constraints():
    """
    Get the constraint matrix for the AML Matek dataset.
    
    The constraint matrix encodes biological relationships between 
    different cell types and abnormalities in the AML Matek dataset.
    
    Returns:
        torch.Tensor: Constraint matrix C ∈ R^(15×15)
    """
    # Class names for reference
    class_names = [
        "Myeloblast", "Promyelocyte", "Myelocyte", "Metamyelocyte",
        "Band Neutrophil", "Segmented Neutrophil", "Eosinophil", "Basophil",
        "Monocyte", "Lymphocyte", "Plasma Cell", "Erythroblast",
        "RBC/Platelet", "Rare/Atypical", "Artifact"
    ]
    
    # Initialize with zeros (no relationship)
    num_classes = 15
    C = np.zeros((num_classes, num_classes))
    
    # Define mutual exclusivity and co-occurrence relationships
    
    # Sequential maturation stages have positive co-occurrence
    # Myeloblast → Promyelocyte → Myelocyte → Metamyelocyte → Band → Segmented
    C[0, 1] = C[1, 0] = 0.3  # Myeloblast and Promyelocyte
    C[1, 2] = C[2, 1] = 0.3  # Promyelocyte and Myelocyte
    C[2, 3] = C[3, 2] = 0.3  # Myelocyte and Metamyelocyte
    C[3, 4] = C[4, 3] = 0.3  # Metamyelocyte and Band
    C[4, 5] = C[5, 4] = 0.3  # Band and Segmented
    
    # Mutual exclusivity between distinct lineages
    # Myeloid cells
    for i in range(6):  # Myeloblast to Segmented Neutrophil
        # Cannot co-exist with lymphoid/other lineages
        C[i, 9] = C[9, i] = -1  # with Lymphocyte
        C[i, 10] = C[10, i] = -1  # with Plasma Cell
        
        # Later myeloid stages cannot co-exist with early ones (skip adjacents)
        if i >= 3:  # Metamyelocyte onward
            C[i, 0] = C[0, i] = -1  # Cannot be Myeloblast
        if i >= 4:  # Band onward
            C[i, 1] = C[1, i] = -1  # Cannot be Promyelocyte
        if i >= 5:  # Segmented onward
            C[i, 2] = C[2, i] = -1  # Cannot be Myelocyte
            C[i, 3] = C[3, i] = -1  # Cannot be Metamyelocyte
    
    # Eosinophil relationships
    C[6, 0] = C[0, 6] = -1  # Eosinophil and Myeloblast
    C[6, 1] = C[1, 6] = 0.2  # Eosinophil and Promyelocyte
    C[6, 2] = C[2, 6] = 0.2  # Eosinophil and Myelocyte
    
    # Basophil relationships
    C[7, 0] = C[0, 7] = -1  # Basophil and Myeloblast
    C[7, 1] = C[1, 7] = 0.2  # Basophil and Promyelocyte
    C[7, 2] = C[2, 7] = 0.2  # Basophil and Myelocyte
    
    # Monocyte relationships
    C[8, 0] = C[0, 8] = 0.2  # Monocyte and Myeloblast
    
    # Lymphocyte and Plasma cell relationship
    C[9, 10] = C[10, 9] = 0.2  # Lymphocyte and Plasma cell
    
    # Erythroblast relationships
    C[11, 12] = C[12, 11] = 0.3  # Erythroblast and RBC/Platelet
    
    # Rare/Atypical weak correlation with all cell types
    for i in range(num_classes - 2):  # All except Artifact
        C[13, i] = C[i, 13] = 0.2
    
    # Artifact can weakly correlate with Rare/Atypical
    C[14, 13] = C[13, 14] = 0.2
    
    return torch.tensor(C, dtype=torch.float32)


def get_bmc_constraints():
    """
    Get the constraint matrix for the BMC (Bone Marrow Cell) dataset.
    
    The constraint matrix encodes biological relationships between 
    different cell types in bone marrow.
    
    Returns:
        torch.Tensor: Constraint matrix C ∈ R^(21×21)
    """
    # Class names for reference
    class_names = [
        "Myeloblast", "Promyelocyte", "Myelocyte", "Metamyelocyte",
        "Band Neutrophil", "Segmented Neutrophil", "Eosinophil", "Basophil",
        "Monocyte", "Lymphocyte", "Plasma Cell", "Erythroblast",
        "Megakaryocyte", "Pro-Erythroblast", "Baso-Erythroblast",
        "Poly-Erythroblast", "Ortho-Erythroblast", "RBC",
        "Artifact", "Smudge", "Other"
    ]
    
    # Initialize with zeros (no relationship)
    num_classes = 21
    C = np.zeros((num_classes, num_classes))
    
    # Define mutual exclusivity and co-occurrence relationships
    
    # Sequential maturation stages have positive co-occurrence
    # Myeloid lineage: Myeloblast → Promyelocyte → Myelocyte → Metamyelocyte → Band → Segmented
    C[0, 1] = C[1, 0] = 0.3  # Myeloblast and Promyelocyte
    C[1, 2] = C[2, 1] = 0.3  # Promyelocyte and Myelocyte
    C[2, 3] = C[3, 2] = 0.3  # Myelocyte and Metamyelocyte
    C[3, 4] = C[4, 3] = 0.3  # Metamyelocyte and Band
    C[4, 5] = C[5, 4] = 0.3  # Band and Segmented
    
    # Erythroid lineage: Pro-Erythro → Baso-Erythro → Poly-Erythro → Ortho-Erythro → RBC
    C[13, 14] = C[14, 13] = 0.3  # Pro-Erythroblast and Baso-Erythroblast
    C[14, 15] = C[15, 14] = 0.3  # Baso-Erythroblast and Poly-Erythroblast
    C[15, 16] = C[16, 15] = 0.3  # Poly-Erythroblast and Ortho-Erythroblast
    C[16, 17] = C[17, 16] = 0.3  # Ortho-Erythroblast and RBC
    
    # Mutual exclusivity between distinct lineages
    # Myeloid with Lymphoid/Erythroid
    for i in range(6):  # Myeloblast to Segmented Neutrophil
        # Cannot co-exist with lymphoid lineage
        C[i, 9] = C[9, i] = -1  # with Lymphocyte
        C[i, 10] = C[10, i] = -1  # with Plasma Cell
        
        # Later myeloid stages cannot co-exist with erythroid lineage
        if i >= 4:  # Band and Segmented
            for j in range(13, 18):  # All erythroid stages
                C[i, j] = C[j, i] = -1
    
    # Eosinophil relationships
    C[6, 0] = C[0, 6] = -1  # Eosinophil and Myeloblast
    C[6, 1] = C[1, 6] = 0.2  # Eosinophil and Promyelocyte
    C[6, 2] = C[2, 6] = 0.2  # Eosinophil and Myelocyte
    
    # Basophil relationships
    C[7, 0] = C[0, 7] = -1  # Basophil and Myeloblast
    C[7, 1] = C[1, 7] = 0.2  # Basophil and Promyelocyte
    C[7, 2] = C[2, 7] = 0.2  # Basophil and Myelocyte
    
    # Monocyte relationships
    C[8, 0] = C[0, 8] = 0.2  # Monocyte and Myeloblast
    
    # Lymphocyte and Plasma cell relationship
    C[9, 10] = C[10, 9] = 0.2  # Lymphocyte and Plasma cell
    
    # Erythroblast general class relationships
    C[11, 13] = C[13, 11] = 0.3  # Erythroblast and Pro-Erythroblast
    C[11, 14] = C[14, 11] = 0.3  # Erythroblast and Baso-Erythroblast
    C[11, 15] = C[15, 11] = 0.3  # Erythroblast and Poly-Erythroblast
    C[11, 16] = C[16, 11] = 0.3  # Erythroblast and Ortho-Erythroblast
    C[11, 17] = C[17, 11] = 0.3  # Erythroblast and RBC
    
    # Megakaryocyte relationships
    C[12, 17] = C[17, 12] = 0.2  # Megakaryocyte and RBC
    
    # Artifact, Smudge, and Other relationships
    C[18, 19] = C[19, 18] = 0.2  # Artifact and Smudge
    C[18, 20] = C[20, 18] = 0.2  # Artifact and Other
    C[19, 20] = C[20, 19] = 0.2  # Smudge and Other
    
    # Smudge cell weak associations with blast cells and lymphocytes
    C[19, 0] = C[0, 19] = 0.15  # Smudge and Myeloblast
    C[19, 9] = C[9, 19] = 0.15  # Smudge and Lymphocyte
    
    return torch.tensor(C, dtype=torch.float32)


def get_constraint_matrix(dataset_name):
    """
    Factory function to get the appropriate constraint matrix.
    
    Args:
        dataset_name (str): Name of the dataset
        
    Returns:
        torch.Tensor: Constraint matrix for the dataset
    """
    dataset_name = dataset_name.lower()
    
    if dataset_name == 'gr_neutro':
        return get_gr_neutro_constraints()
    elif dataset_name == 'aml_matek':
        return get_aml_matek_constraints()
    elif dataset_name == 'bmc':
        return get_bmc_constraints()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def save_constraint_matrix(dataset_name, output_path):
    """
    Save the constraint matrix for a dataset to a file.
    
    Args:
        dataset_name (str): Name of the dataset
        output_path (str): Path to save the constraint matrix
    """
    constraint_matrix = get_constraint_matrix(dataset_name)
    torch.save(constraint_matrix, output_path)
    print(f"Constraint matrix for {dataset_name} saved to {output_path}")
