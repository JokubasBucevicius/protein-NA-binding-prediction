"""
Generate multiclass binding site predictions for dual DNA/RNA binding proteins.

This script generates predictions for proteins that may bind both DNA and RNA,
outputting separate probabilities for each class.

Output PDB files use:
- B-factor: DNA binding probability
- Occupancy: RNA binding probability

Usage:
    cd /home/jokubas/Magistras/duomenys/visualizations
    python generate_multiclass_predictions.py
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

MODEL_PATH = os.path.join(SCRIPT_DIR, "Joint_MLPGNN_multiclass_weight0.2.pt")
GRAPH_DIR = os.path.join(BASE_DIR, "grafai", "cas9")
EMB_DIR = os.path.join(BASE_DIR, "emb", "ESM2", "cas9")

# Proteins to process - dual binding proteins
PROTEINS = {
    "multiclass": {
        "class": "dgDNR",  # Use dgDNR class for graph data
        "proteins": ["9lgi_A"]
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
        'input_dim': 37,
        'hidden_dim': 128,
        'heads': 4,
        'dropout': 0.3
    },
    'mlp_loss_weight': 0.2
}


# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

class BindingSiteGNN(nn.Module):
    """GNN for per-residue binding site prediction using GATv2 attention."""

    def __init__(self, input_dim=5, hidden_dim=128, heads=4, dropout=0.2,
                 num_amino_acids=20, aa_embed_dim=32, output_dim=2):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.aa_embedding = nn.Embedding(num_amino_acids, aa_embed_dim)
        total_input_dim = input_dim + aa_embed_dim

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

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        residue_type = data.residue_type

        aa_emb = self.aa_embedding(residue_type)
        x = torch.cat([x, aa_emb], dim=-1)

        x = self.conv1(x, edge_index, edge_attr)
        x = F.elu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index, edge_attr)
        x = F.elu(x)
        x = self.dropout(x)

        logits = self.fc(x)
        return logits


class JointMLPGNNMulti(nn.Module):
    """Joint MLP+GNN for multiclass (3 classes)."""

    def __init__(self, mlp_config, gnn_config, mlp_loss_weight=0.3):
        super().__init__()
        self.mlp_loss_weight = mlp_loss_weight
        self.mlp_embedding_dim = mlp_config['hidden_dims'][-1]

        self.mlp_layers = nn.ModuleList()
        prev_dim = mlp_config['input_dim']
        for hidden_dim in mlp_config['hidden_dims']:
            self.mlp_layers.append(nn.Linear(prev_dim, hidden_dim))
            self.mlp_layers.append(nn.ReLU())
            self.mlp_layers.append(nn.Dropout(mlp_config['dropout']))
            self.mlp_layers.append(nn.LayerNorm(hidden_dim))
            prev_dim = hidden_dim

        self.mlp_head = nn.Linear(prev_dim, mlp_config['output_dim'])

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
# DATA LOADING FUNCTIONS
# =============================================================================

def normalize_column(values):
    values = np.array(values, dtype=np.float32)
    min_val, max_val = values.min(), values.max()
    if max_val - min_val > 0:
        return (values - min_val) / (max_val - min_val)
    return np.zeros_like(values)


def normalize_voromqa(values):
    values = np.array(values, dtype=np.float32)
    if values.min() < 0:
        values = values - values.min()
    values = np.log1p(values)
    return normalize_column(values)


def aggregate_atoms_to_residues(nodes_df):
    pooling = {
        "residue_type": "first",
        "sas_area": "sum",
        "ev14": "max",
        "ev28": "max",
        "ev56": "max",
        "voromqa_sas_energy": "sum",
        "bsite": "max",
    }
    pooling = {k: v for k, v in pooling.items() if k in nodes_df.columns}
    residue_df = nodes_df.groupby("ID_resSeq").agg(pooling).reset_index()
    return residue_df


def aggregate_edges_to_residues(edges_df):
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
    """Load protein data for inference."""
    graph_nodes_path = os.path.join(GRAPH_DIR, class_name,
                                    f"{protein_name}_chain_A", "graph_nodes.csv")
    graph_edges_path = os.path.join(GRAPH_DIR, class_name,
                                    f"{protein_name}_chain_A", "graph_links.csv")
    embedding_path = os.path.join(EMB_DIR, class_name, f"{protein_name}.pt")

    if not os.path.exists(graph_nodes_path):
        print(f"  Warning: Graph nodes not found: {graph_nodes_path}")
        return None, None, None
    if not os.path.exists(embedding_path):
        print(f"  Warning: Embeddings not found: {embedding_path}")
        return None, None, None

    emb_data = torch.load(embedding_path, map_location='cpu', weights_only=False)
    if isinstance(emb_data, dict) and 'representations' in emb_data:
        layer_keys = sorted(emb_data['representations'].keys())
        embeddings = emb_data['representations'][layer_keys[-1]]
    else:
        embeddings = emb_data

    nodes_df = pd.read_csv(graph_nodes_path)
    residue_df = aggregate_atoms_to_residues(nodes_df)

    surface_mask = residue_df['sas_area'] > 0
    residue_df = residue_df[surface_mask].reset_index(drop=True)

    if len(residue_df) == 0:
        print(f"  Warning: No surface residues for {protein_name}")
        return None, None, None

    edges_df = None
    if os.path.exists(graph_edges_path):
        edges_df = pd.read_csv(graph_edges_path)
        edges_df = aggregate_edges_to_residues(edges_df)
        surface_ids = set(residue_df['ID_resSeq'])
        edge_mask = (edges_df['ID1_resSeq'].isin(surface_ids) &
                     edges_df['ID2_resSeq'].isin(surface_ids))
        edges_df = edges_df[edge_mask].reset_index(drop=True)

    id_res_seq_values = residue_df['ID_resSeq'].values
    min_res_seq = id_res_seq_values.min()
    emb_size = embeddings.shape[0]

    if min_res_seq >= 1:
        embedding_indices = id_res_seq_values - 1
    else:
        embedding_indices = id_res_seq_values

    valid_mask = (embedding_indices >= 0) & (embedding_indices < emb_size)
    if valid_mask.sum() == 0:
        print(f"  Warning: No valid embedding indices for {protein_name}")
        return None, None, None

    if valid_mask.sum() < len(embedding_indices):
        print(f"  Warning: Filtering {len(embedding_indices) - valid_mask.sum()} invalid residues")
        embedding_indices = embedding_indices[valid_mask]
        residue_df = residue_df[valid_mask].reset_index(drop=True)

    esm_embeddings = embeddings[embedding_indices, :]

    structural_features = np.stack([
        normalize_column(residue_df['sas_area'].values),
        normalize_voromqa(residue_df['voromqa_sas_energy'].values),
        normalize_column(residue_df['ev14'].values),
        normalize_column(residue_df['ev28'].values),
        normalize_column(residue_df['ev56'].values),
    ], axis=1).astype(np.float32)

    residue_types = torch.tensor(residue_df['residue_type'].values, dtype=torch.long)

    if edges_df is not None and len(edges_df) > 0:
        id_to_idx = {id_val: idx for idx, id_val in enumerate(residue_df['ID_resSeq'])}
        src_indices = [id_to_idx.get(id1, -1) for id1 in edges_df['ID1_resSeq']]
        dst_indices = [id_to_idx.get(id2, -1) for id2 in edges_df['ID2_resSeq']]

        valid_edges = [(s, d) for s, d in zip(src_indices, dst_indices) if s >= 0 and d >= 0]
        if valid_edges:
            src_indices, dst_indices = zip(*valid_edges)
            edge_index = torch.tensor([list(src_indices), list(dst_indices)], dtype=torch.long)

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

    graph_data = Data(
        x=torch.tensor(structural_features, dtype=torch.float32),
        residue_type=residue_types,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=len(residue_df)
    )

    residue_ids = residue_df['ID_resSeq'].tolist()

    return esm_embeddings, graph_data, residue_ids


# =============================================================================
# GROUND TRUTH EXTRACTION
# =============================================================================

def extract_ground_truth_from_pdb(pdb_path, contact_distance=4.5):
    """
    Extract ground truth binding labels from PDB by finding protein-nucleic acid contacts.

    Args:
        pdb_path: Path to full PDB file with protein and nucleic acids
        contact_distance: Distance cutoff for contacts in Angstroms

    Returns:
        dna_binding: Dict mapping residue number to 1.0 if binds DNA, 0.0 otherwise
        rna_binding: Dict mapping residue number to 1.0 if binds RNA, 0.0 otherwise
    """
    # DNA and RNA residue names
    dna_residues = {'DA', 'DT', 'DC', 'DG', 'DU'}
    rna_residues = {'A', 'U', 'C', 'G'}  # Single letter for RNA

    # Standard amino acids
    amino_acids = {
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
        'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
    }

    protein_atoms = []  # (res_num, x, y, z)
    dna_atoms = []      # (x, y, z)
    rna_atoms = []      # (x, y, z)

    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                res_name = line[17:20].strip()
                try:
                    res_num = int(line[22:26].strip())
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())

                    if res_name in amino_acids:
                        protein_atoms.append((res_num, x, y, z))
                    elif res_name in dna_residues:
                        dna_atoms.append((x, y, z))
                    elif res_name in rna_residues:
                        rna_atoms.append((x, y, z))
                except (ValueError, IndexError):
                    continue

    print(f"    Found {len(protein_atoms)} protein atoms, {len(dna_atoms)} DNA atoms, {len(rna_atoms)} RNA atoms")

    # Find contacts
    dna_binding_residues = set()
    rna_binding_residues = set()
    cutoff_sq = contact_distance ** 2

    # Convert to numpy for faster computation
    if protein_atoms:
        protein_arr = np.array([(a[1], a[2], a[3]) for a in protein_atoms])
        protein_res = np.array([a[0] for a in protein_atoms])

        # Check DNA contacts
        if dna_atoms:
            dna_arr = np.array(dna_atoms)
            for i, (px, py, pz) in enumerate(protein_arr):
                dists_sq = (dna_arr[:, 0] - px)**2 + (dna_arr[:, 1] - py)**2 + (dna_arr[:, 2] - pz)**2
                if np.any(dists_sq <= cutoff_sq):
                    dna_binding_residues.add(protein_res[i])

        # Check RNA contacts
        if rna_atoms:
            rna_arr = np.array(rna_atoms)
            for i, (px, py, pz) in enumerate(protein_arr):
                dists_sq = (rna_arr[:, 0] - px)**2 + (rna_arr[:, 1] - py)**2 + (rna_arr[:, 2] - pz)**2
                if np.any(dists_sq <= cutoff_sq):
                    rna_binding_residues.add(protein_res[i])

    # Get all protein residues
    all_protein_residues = set(a[0] for a in protein_atoms)

    # Create binding dictionaries
    dna_binding = {res: 1.0 if res in dna_binding_residues else 0.0 for res in all_protein_residues}
    rna_binding = {res: 1.0 if res in rna_binding_residues else 0.0 for res in all_protein_residues}

    print(f"    Ground truth: {len(dna_binding_residues)} DNA-binding, {len(rna_binding_residues)} RNA-binding residues")
    both_count = len(dna_binding_residues & rna_binding_residues)
    if both_count > 0:
        print(f"    Both DNA+RNA binding: {both_count} residues")

    return dna_binding, rna_binding


# =============================================================================
# PDB MODIFICATION FUNCTIONS
# =============================================================================

def modify_pdb_multiclass(input_pdb_path, output_pdb_path, dna_probs, rna_probs):
    """
    Modify PDB file with multiclass probabilities.

    B-factor: DNA binding probability
    Occupancy: RNA binding probability
    Nucleic acid atoms are removed.
    """
    nucleic_acid_residues = {
        'DA', 'DT', 'DC', 'DG', 'DU',
        'A', 'T', 'C', 'G', 'U',
        'RA', 'RU', 'RC', 'RG',
        'ADE', 'THY', 'CYT', 'GUA', 'URA',
    }

    with open(input_pdb_path, 'r') as f:
        lines = f.readlines()

    modified_lines = []
    for line in lines:
        if line.startswith('ATOM') or line.startswith('HETATM'):
            res_name = line[17:20].strip()
            if res_name in nucleic_acid_residues:
                continue

            try:
                res_num = int(line[22:26].strip())
                dna_prob = dna_probs.get(res_num, 0.0)
                rna_prob = rna_probs.get(res_num, 0.0)

                # Ensure line is long enough (pad with spaces if needed)
                line_content = line.rstrip('\n')
                if len(line_content) < 80:
                    line_content = line_content.ljust(80)

                # PDB format (1-indexed):
                # Occupancy: columns 55-60 (0-indexed: 54-60)
                # B-factor: columns 61-66 (0-indexed: 60-66)
                occupancy_str = f"{rna_prob:6.3f}"  # Use 3 decimals for small RNA values
                bfactor_str = f"{dna_prob:6.3f}"

                modified_line = line_content[:54] + occupancy_str + bfactor_str + line_content[66:] + '\n'
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
    print("Multiclass Binding Site Prediction")
    print("DNA (class 1) + RNA (class 2)")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    print(f"\nLoading model from: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found: {MODEL_PATH}")
        return

    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

    model = JointMLPGNNMulti(
        mlp_config=MODEL_CONFIG['mlp'],
        gnn_config=MODEL_CONFIG['gnn'],
        mlp_loss_weight=MODEL_CONFIG['mlp_loss_weight']
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print("Model loaded successfully!")

    for folder, config in PROTEINS.items():
        class_name = config['class']
        proteins = config['proteins']

        print(f"\n{'='*60}")
        print(f"Processing {folder} ({class_name})")
        print(f"{'='*60}")

        # Output to multiclass folder
        folder_path = os.path.join(SCRIPT_DIR, "multiclass")
        os.makedirs(folder_path, exist_ok=True)

        for protein_name in proteins:
            print(f"\n  Processing: {protein_name}")

            esm_emb, graph_data, residue_ids = load_protein_data(protein_name, class_name)

            if esm_emb is None:
                print(f"    Skipping {protein_name} - could not load data")
                continue

            print(f"    Loaded {len(residue_ids)} surface residues")

            esm_emb = esm_emb.to(device)
            graph_data = graph_data.to(device)

            with torch.no_grad():
                gnn_logits, mlp_logits = model(esm_emb, graph_data)
                probs = F.softmax(gnn_logits, dim=-1)

                # Class 0: Non-binding, Class 1: DNA, Class 2: RNA
                non_binding_probs = probs[:, 0].cpu().numpy()
                dna_probs_arr = probs[:, 1].cpu().numpy()
                rna_probs_arr = probs[:, 2].cpu().numpy()

            # Create residue -> probability mappings
            dna_probs = {res_id: float(prob) for res_id, prob in zip(residue_ids, dna_probs_arr)}
            rna_probs = {res_id: float(prob) for res_id, prob in zip(residue_ids, rna_probs_arr)}

            print(f"    DNA probability range: {dna_probs_arr.min():.3f} - {dna_probs_arr.max():.3f}")
            print(f"    RNA probability range: {rna_probs_arr.min():.3f} - {rna_probs_arr.max():.3f}")

            # Count residues above threshold
            dna_threshold = 0.3
            rna_threshold = 0.3
            dna_count = sum(1 for p in dna_probs_arr if p > dna_threshold)
            rna_count = sum(1 for p in rna_probs_arr if p > rna_threshold)
            both_count = sum(1 for d, r in zip(dna_probs_arr, rna_probs_arr)
                           if d > dna_threshold and r > rna_threshold)

            print(f"    DNA binding (>{dna_threshold}): {dna_count} residues")
            print(f"    RNA binding (>{rna_threshold}): {rna_count} residues")
            print(f"    Both DNA+RNA: {both_count} residues")

            # Find input PDB file (protein only)
            input_pdb = os.path.join(SCRIPT_DIR, f"{protein_name}.pdb")
            if not os.path.exists(input_pdb):
                print(f"    Warning: PDB file not found: {input_pdb}")
                continue

            # Modify PDB with multiclass predictions
            output_pdb = os.path.join(folder_path, f"{protein_name}_multiclass.pdb")
            modify_pdb_multiclass(input_pdb, output_pdb, dna_probs, rna_probs)
            print(f"    Saved: {output_pdb}")

            # Extract and save ground truth from full PDB (with nucleic acids)
            full_pdb = os.path.join(SCRIPT_DIR, f"{protein_name}_visas.pdb")
            if os.path.exists(full_pdb):
                print(f"\n    Extracting ground truth from: {full_pdb}")
                gt_dna, gt_rna = extract_ground_truth_from_pdb(full_pdb)

                # Save ground truth PDB
                gt_output_pdb = os.path.join(folder_path, f"{protein_name}_ground_truth.pdb")
                modify_pdb_multiclass(input_pdb, gt_output_pdb, gt_dna, gt_rna)
                print(f"    Saved: {gt_output_pdb}")
            else:
                print(f"    Warning: Full PDB not found: {full_pdb} (no ground truth generated)")

    print(f"\n{'='*60}")
    print("Done! Multiclass PDB files have been generated.")
    print("Run 'pymol visualize_multiclass.pml' to visualize.")
    print("=" * 60)


if __name__ == "__main__":
    main()
