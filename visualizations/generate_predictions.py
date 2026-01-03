"""
Generate binding site predictions and modify PDB files with probabilities as B-factors.

This script:
1. Loads the trained Joint MLP+GNN model
2. For each protein: loads embeddings + graph, runs inference
3. Modifies PDB files to replace B-factors with binding probabilities

Usage:
    cd /home/jokubas/Magistras/duomenys/visualizations
    python generate_predictions.py
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv


# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths (relative to this script's location)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)  # /home/jokubas/Magistras/duomenys

MODEL_PATH = os.path.join(SCRIPT_DIR, "Joint_MLPGNN_multiclass_weight0.2.pt")
GRAPH_DIR = os.path.join(BASE_DIR, "grafai", "graphs", "k95")
EMB_DIR = os.path.join(BASE_DIR, "emb", "ESM2", "650M")

# Proteins to process
PROTEINS = {
    # DNA-binding proteins (dgDNR class)
    "dna-proteins": {
        "class": "dgDNR",
        "proteins": ["1b72_A", "1bl0_A", "1c8c_A", "1le8_A", "1llm_D"]
    },
    # RNA-binding proteins (RNR class)
    "rna-proteins": {
        "class": "RNR",
        "proteins": ["1qtq_A", "2db3_A", "2i82_A", "2jlv_A", "3bso_A"]
    }
}

# Model configuration (must match training)
MODEL_CONFIG = {
    'mlp': {
        'input_dim': 1280,
        'hidden_dims': [256, 64, 32],
        'output_dim': 3,
        'dropout': 0.3
    },
    'gnn': {
        'input_dim': 37,  # 5 structural + 32 MLP embedding
        'hidden_dim': 128,
        'heads': 4,
        'dropout': 0.3
    },
    'mlp_loss_weight': 0.2
}


# =============================================================================
# MODEL DEFINITIONS (from main-workflow.ipynb)
# =============================================================================

class BindingSiteGNN(nn.Module):
    """GNN for per-residue binding site prediction using GATv2 attention."""

    def __init__(self, input_dim=5, hidden_dim=128, heads=4, dropout=0.2,
                 num_amino_acids=20, aa_embed_dim=32, output_dim=2):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        # Learnable amino acid embeddings
        self.aa_embedding = nn.Embedding(num_amino_acids, aa_embed_dim)

        # Total input = structural features + AA embedding
        total_input_dim = input_dim + aa_embed_dim

        # GATv2 layers
        self.conv1 = GATv2Conv(
            in_channels=total_input_dim,
            out_channels=hidden_dim,
            heads=heads,
            edge_dim=2,
        )

        self.conv2 = GATv2Conv(
            in_channels=hidden_dim * heads,
            out_channels=hidden_dim,
            heads=1,
            edge_dim=2,
        )

        # Output layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        residue_type = data.residue_type

        # Get amino acid embeddings and concatenate
        aa_emb = self.aa_embedding(residue_type)
        x = torch.cat([x, aa_emb], dim=-1)

        # GNN layers
        x = self.conv1(x, edge_index, edge_attr)
        x = F.elu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index, edge_attr)
        x = F.elu(x)
        x = self.dropout(x)

        # Output
        logits = self.fc(x)
        return logits


class JointMLPGNNMulti(nn.Module):
    """Joint MLP+GNN for multiclass (3 classes)."""

    def __init__(self, mlp_config, gnn_config, mlp_loss_weight=0.3):
        super().__init__()
        self.mlp_loss_weight = mlp_loss_weight
        self.mlp_embedding_dim = mlp_config['hidden_dims'][-1]

        # MLP layers
        self.mlp_layers = nn.ModuleList()
        prev_dim = mlp_config['input_dim']
        for hidden_dim in mlp_config['hidden_dims']:
            self.mlp_layers.append(nn.Linear(prev_dim, hidden_dim))
            self.mlp_layers.append(nn.ReLU())
            self.mlp_layers.append(nn.Dropout(mlp_config['dropout']))
            self.mlp_layers.append(nn.LayerNorm(hidden_dim))
            prev_dim = hidden_dim

        self.mlp_head = nn.Linear(prev_dim, mlp_config['output_dim'])

        # GNN
        self.gnn = BindingSiteGNN(
            input_dim=gnn_config['input_dim'],
            hidden_dim=gnn_config['hidden_dim'],
            heads=gnn_config['heads'],
            dropout=gnn_config['dropout'],
            output_dim=3,
        )

    def mlp_forward(self, x):
        for layer in self.mlp_layers:
            x = layer(x)
        embeddings = x
        logits = self.mlp_head(embeddings)
        return logits, embeddings

    def forward(self, esm_embeddings, graph_data):
        mlp_logits, mlp_embeddings = self.mlp_forward(esm_embeddings)
        combined_features = torch.cat([graph_data.x, mlp_embeddings], dim=-1)
        graph_data.x = combined_features
        gnn_logits = self.gnn(graph_data)
        return gnn_logits, mlp_logits


# =============================================================================
# DATA LOADING FUNCTIONS (from main-workflow.ipynb)
# =============================================================================

def normalize_column(values):
    """Min-max normalization to [0, 1]."""
    values = np.array(values, dtype=np.float32)
    min_val, max_val = values.min(), values.max()
    if max_val - min_val > 0:
        return (values - min_val) / (max_val - min_val)
    return np.zeros_like(values)


def normalize_voromqa(values):
    """Specialized normalization for voromqa_sas_energy."""
    values = np.array(values, dtype=np.float32)
    if values.min() < 0:
        values = values - values.min()
    values = np.log1p(values)
    return normalize_column(values)


def aggregate_atoms_to_residues(nodes_df):
    """Aggregate atom-level features to residue-level."""
    pooling = {
        "residue_type": "first",
        "sas_area": "sum",
        "ev14": "max",
        "ev28": "max",
        "ev56": "max",
        "voromqa_sas_energy": "sum",
        "bsite": "max",
    }

    # Only include columns that exist
    pooling = {k: v for k, v in pooling.items() if k in nodes_df.columns}

    residue_df = nodes_df.groupby("ID_resSeq").agg(pooling).reset_index()
    return residue_df


def aggregate_edges_to_residues(edges_df):
    """Aggregate atom-level edges to residue-level edges."""
    edges_df = edges_df[edges_df["ID1_resSeq"] != edges_df["ID2_resSeq"]].copy()
    edges_df["res1"] = edges_df[["ID1_resSeq", "ID2_resSeq"]].min(axis=1)
    edges_df["res2"] = edges_df[["ID1_resSeq", "ID2_resSeq"]].max(axis=1)

    residue_edges = edges_df.groupby(["res1", "res2"]).agg({
        "area": "sum",
        "boundary": "mean"
    }).reset_index()

    residue_edges = residue_edges.rename(columns={
        "res1": "ID1_resSeq",
        "res2": "ID2_resSeq"
    })
    return residue_edges


def load_protein_data(protein_name, class_name):
    """
    Load protein data for inference.

    Returns:
        esm_embeddings: Tensor [N, 1280] - ESM embeddings for surface residues
        graph_data: PyG Data object with structural features
        residue_ids: List of ID_resSeq values for mapping back to PDB
        ground_truth: Dict mapping residue ID to binding site label (0 or 1)
    """
    # Construct paths
    graph_nodes_path = os.path.join(GRAPH_DIR, class_name,
                                    f"{protein_name}_chain_A", "graph_nodes.csv")
    graph_edges_path = os.path.join(GRAPH_DIR, class_name,
                                    f"{protein_name}_chain_A", "graph_links.csv")
    embedding_path = os.path.join(EMB_DIR, class_name, f"{protein_name}.pt")

    # Check files exist
    if not os.path.exists(graph_nodes_path):
        print(f"  Warning: Graph nodes not found: {graph_nodes_path}")
        return None, None, None, None
    if not os.path.exists(embedding_path):
        print(f"  Warning: Embeddings not found: {embedding_path}")
        return None, None, None, None

    # Load embeddings
    emb_data = torch.load(embedding_path, map_location='cpu', weights_only=False)
    if isinstance(emb_data, dict) and 'representations' in emb_data:
        layer_keys = sorted(emb_data['representations'].keys())
        embeddings = emb_data['representations'][layer_keys[-1]]
    else:
        embeddings = emb_data

    # Load graph nodes
    nodes_df = pd.read_csv(graph_nodes_path)
    residue_df = aggregate_atoms_to_residues(nodes_df)

    # Filter surface residues
    surface_mask = residue_df['sas_area'] > 0
    residue_df = residue_df[surface_mask].reset_index(drop=True)

    if len(residue_df) == 0:
        print(f"  Warning: No surface residues for {protein_name}")
        return None, None, None, None

    # Load and process edges
    edges_df = None
    if os.path.exists(graph_edges_path):
        edges_df = pd.read_csv(graph_edges_path)
        edges_df = aggregate_edges_to_residues(edges_df)
        # Filter to surface residues only
        surface_ids = set(residue_df['ID_resSeq'])
        edge_mask = (edges_df['ID1_resSeq'].isin(surface_ids) &
                     edges_df['ID2_resSeq'].isin(surface_ids))
        edges_df = edges_df[edge_mask].reset_index(drop=True)

    # Map ID_resSeq to embedding indices
    id_res_seq_values = residue_df['ID_resSeq'].values
    min_res_seq = id_res_seq_values.min()
    emb_size = embeddings.shape[0]

    if min_res_seq >= 1:
        embedding_indices = id_res_seq_values - 1
    else:
        embedding_indices = id_res_seq_values

    # Validate indices
    valid_mask = (embedding_indices >= 0) & (embedding_indices < emb_size)
    if valid_mask.sum() == 0:
        print(f"  Warning: No valid embedding indices for {protein_name}")
        return None, None, None, None

    if valid_mask.sum() < len(embedding_indices):
        print(f"  Warning: Filtering {len(embedding_indices) - valid_mask.sum()} invalid residues")
        embedding_indices = embedding_indices[valid_mask]
        residue_df = residue_df[valid_mask].reset_index(drop=True)

    # Get ESM embeddings for surface residues
    esm_embeddings = embeddings[embedding_indices, :]

    # Create structural features (5 dims)
    structural_features = np.stack([
        normalize_column(residue_df['sas_area'].values),
        normalize_voromqa(residue_df['voromqa_sas_energy'].values),
        normalize_column(residue_df['ev14'].values),
        normalize_column(residue_df['ev28'].values),
        normalize_column(residue_df['ev56'].values),
    ], axis=1).astype(np.float32)

    # Residue types
    residue_types = torch.tensor(residue_df['residue_type'].values, dtype=torch.long)

    # Edge index and attributes
    if edges_df is not None and len(edges_df) > 0:
        id_to_idx = {id_val: idx for idx, id_val in enumerate(residue_df['ID_resSeq'])}
        src_indices = [id_to_idx.get(id1, -1) for id1 in edges_df['ID1_resSeq']]
        dst_indices = [id_to_idx.get(id2, -1) for id2 in edges_df['ID2_resSeq']]

        # Filter valid edges
        valid_edges = [(s, d) for s, d in zip(src_indices, dst_indices) if s >= 0 and d >= 0]
        if valid_edges:
            src_indices, dst_indices = zip(*valid_edges)
            edge_index = torch.tensor([list(src_indices), list(dst_indices)], dtype=torch.long)

            # Get edge attributes for valid edges
            valid_edge_mask = [(id_to_idx.get(id1, -1) >= 0 and id_to_idx.get(id2, -1) >= 0)
                               for id1, id2 in zip(edges_df['ID1_resSeq'], edges_df['ID2_resSeq'])]
            edge_attr = torch.tensor(
                edges_df.loc[valid_edge_mask, ['area', 'boundary']].values, dtype=torch.float32
            )
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 2), dtype=torch.float32)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 2), dtype=torch.float32)

    # Create PyG Data object
    graph_data = Data(
        x=torch.tensor(structural_features, dtype=torch.float32),
        residue_type=residue_types,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=len(residue_df)
    )

    # Return residue IDs for mapping back to PDB
    residue_ids = residue_df['ID_resSeq'].tolist()

    # Extract ground truth binding site labels
    ground_truth = {}
    if 'bsite' in residue_df.columns:
        ground_truth = {res_id: float(bsite) for res_id, bsite in
                        zip(residue_df['ID_resSeq'], residue_df['bsite'])}

    return esm_embeddings, graph_data, residue_ids, ground_truth


# =============================================================================
# PDB MODIFICATION FUNCTIONS
# =============================================================================

def modify_pdb_bfactor(input_pdb_path, output_pdb_path, residue_probs):
    """
    Modify PDB file to replace B-factors with binding probabilities.
    Nucleic acid atoms are removed from the output.

    Args:
        input_pdb_path: Path to original PDB file
        output_pdb_path: Path to save modified PDB
        residue_probs: Dict mapping residue number (ID_resSeq) to probability
    """
    # Nucleic acid residue names to exclude
    nucleic_acid_residues = {
        'DA', 'DT', 'DC', 'DG', 'DU',  # DNA
        'A', 'T', 'C', 'G', 'U',        # Single letter (context-dependent)
        'RA', 'RU', 'RC', 'RG',         # RNA
        'ADE', 'THY', 'CYT', 'GUA', 'URA',  # Full names
    }

    with open(input_pdb_path, 'r') as f:
        lines = f.readlines()

    modified_lines = []
    for line in lines:
        if line.startswith('ATOM') or line.startswith('HETATM'):
            # Check if this is a nucleic acid residue (columns 18-20, 0-indexed: 17-20)
            res_name = line[17:20].strip()
            if res_name in nucleic_acid_residues:
                continue  # Skip nucleic acid atoms
            # PDB format: columns are 1-indexed
            # Residue number: columns 23-26 (0-indexed: 22-26)
            # B-factor: columns 61-66 (0-indexed: 60-66)
            try:
                res_num = int(line[22:26].strip())
                prob = residue_probs.get(res_num, 0.0)

                # Format probability as B-factor (6 characters, 2 decimal places)
                bfactor_str = f"{prob:6.2f}"

                # Replace B-factor in line
                modified_line = line[:60] + bfactor_str + line[66:]
                modified_lines.append(modified_line)
            except (ValueError, IndexError):
                modified_lines.append(line)
        else:
            modified_lines.append(line)

    with open(output_pdb_path, 'w') as f:
        f.writelines(modified_lines)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print("=" * 60)
    print("Binding Site Prediction - PDB Generation")
    print("=" * 60)

    # Check for CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Load model
    print(f"\nLoading model from: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found: {MODEL_PATH}")
        return

    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

    # Create model
    model = JointMLPGNNMulti(
        mlp_config=MODEL_CONFIG['mlp'],
        gnn_config=MODEL_CONFIG['gnn'],
        mlp_loss_weight=MODEL_CONFIG['mlp_loss_weight']
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print("Model loaded successfully!")

    # Process each protein group
    for folder, config in PROTEINS.items():
        class_name = config['class']
        proteins = config['proteins']

        print(f"\n{'='*60}")
        print(f"Processing {folder} ({class_name})")
        print(f"{'='*60}")

        folder_path = os.path.join(SCRIPT_DIR, folder)

        for protein_name in proteins:
            print(f"\n  Processing: {protein_name}")

            # Load data
            esm_emb, graph_data, residue_ids, ground_truth = load_protein_data(protein_name, class_name)

            if esm_emb is None:
                print(f"    Skipping {protein_name} - could not load data")
                continue

            print(f"    Loaded {len(residue_ids)} surface residues")
            if ground_truth:
                binding_count = sum(1 for v in ground_truth.values() if v > 0)
                print(f"    Ground truth: {binding_count} binding residues")

            # Move to device
            esm_emb = esm_emb.to(device)
            graph_data = graph_data.to(device)

            # Run inference
            with torch.no_grad():
                gnn_logits, mlp_logits = model(esm_emb, graph_data)

                # Convert to binding probabilities
                # For multiclass: prob(binding) = prob(DNA) + prob(RNA)
                probs = F.softmax(gnn_logits, dim=-1)
                binding_probs = (probs[:, 1] + probs[:, 2]).cpu().numpy()

            # Create residue -> probability mapping
            residue_probs = {res_id: float(prob) for res_id, prob in zip(residue_ids, binding_probs)}

            print(f"    Probability range: {binding_probs.min():.3f} - {binding_probs.max():.3f}")

            # Find input PDB file
            input_pdb = os.path.join(folder_path, f"{protein_name}.pdb")
            if not os.path.exists(input_pdb):
                print(f"    Warning: PDB file not found: {input_pdb}")
                continue

            # Modify PDB with predictions
            output_pdb = os.path.join(folder_path, f"{protein_name}_predicted.pdb")
            modify_pdb_bfactor(input_pdb, output_pdb, residue_probs)
            print(f"    Saved: {output_pdb}")

            # Modify PDB with ground truth
            if ground_truth:
                gt_output_pdb = os.path.join(folder_path, f"{protein_name}_ground_truth.pdb")
                modify_pdb_bfactor(input_pdb, gt_output_pdb, ground_truth)
                print(f"    Saved: {gt_output_pdb}")

    print(f"\n{'='*60}")
    print("Done! PDB files have been generated.")
    print("Run 'pymol visualize_binding.pml' to visualize predictions.")
    print("Run 'pymol visualize_ground_truth.pml' to visualize ground truth.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
