"""
Affinity Training Featurizer

This module contains simplified featurizer for affinity training,
with direct PDB and SDF parsing capabilities.
"""

import numpy as np
import torch
from torch_geometric.utils import dense_to_sparse
from rdkit import Chem
from typing import Tuple, List, Optional
from .parser import BOND_PATTERN_TABLE
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

LIGAND_BOND_TYPES = {
    Chem.rdchem.BondType.SINGLE: 0,
    Chem.rdchem.BondType.DOUBLE: 1,
    Chem.rdchem.BondType.TRIPLE: 2,
    Chem.rdchem.BondType.AROMATIC: 3,
}

# Edge types (1-based)
EDGE_TYPES = {
    # Global edges (highest priority)
    'global': 1,               # Global node edges
    
    # Protein-protein edges
    'protein_sequence': 2,     # Sequence-adjacent residues
    'protein_spatial': 3,       # Spatially close residues
    'protein_ligand': 4,        # Protein-ligand interactions
    'protein_covalent': 5,      # Covalent bonds within residues
    
    # Ligand internal edges (using original ligand edge types)
    'ligand_bond_0': 6,        # Ligand bond type 0
    'ligand_bond_1': 7,        # Ligand bond type 1  
    'ligand_bond_2': 8,        # Ligand bond type 2
    'ligand_bond_3': 9,        # Ligand bond type 3
    'ligand_las': 10           # LAS edges (virtual edges)
}

################################
# LAS edges, from FABind_plus
################################
def binarize(x):
    return torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))

def n_hops_adj(adj, n_hops):
    adj_mats = [torch.eye(adj.size(0), dtype=torch.long, device=adj.device), 
    binarize(adj + torch.eye(adj.size(0), dtype=torch.long, device=adj.device))]
    for i in range(2, n_hops+1):
        adj_mats.append(binarize(adj_mats[i-1] @ adj_mats[1]))
    extend_mat = torch.zeros_like(adj)
    for i in range(1, n_hops+1):
        extend_mat += (adj_mats[i] - adj_mats[i-1]) * i
    return extend_mat

def get_LAS_distance_constraint_mask(mol):
    # Get the adj
    adj = Chem.GetAdjacencyMatrix(mol)
    adj = torch.from_numpy(adj)
    extend_adj = n_hops_adj(adj,2)
    # add ring
    ssr = Chem.GetSymmSSSR(mol)
    for ring in ssr:
        for i in ring:
            for j in ring:
                if i==j:
                    continue
                else:
                    extend_adj[i][j]+=1
    # turn to mask
    mol_mask = binarize(extend_adj)
    return mol_mask

def crop_pocket(
    protein_coords: np.ndarray,
    protein_atom_types: np.ndarray,
    protein_residue_types: np.ndarray,
    protein_residue_indices: np.ndarray,
    ligand_coords: np.ndarray,
    max_atoms: int,
    max_residues: int,
    distance_threshold: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Select a residue subset ordered by min distance under budgets.

    The function assumes inputs are already restricted to the original pocket.
    It chooses residues greedily by ascending distance until adding the next
    residue would exceed either the atom budget or the residue budget.
    
    Parameters
    ----------
    distance_threshold : Optional[float]
        Maximum distance from ligand to include residues. If None, no distance filtering is applied.
        If the distance threshold results in fewer than 100 residues, the 100 closest residues are kept instead.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        filtered protein arrays (coords, atom_types, residue_types, residue_indices).
    """
    # Ensure numpy arrays
    protein_coords = np.asarray(protein_coords)
    protein_atom_types = np.asarray(protein_atom_types)
    protein_residue_types = np.asarray(protein_residue_types)
    protein_residue_indices = np.asarray(protein_residue_indices)
    ligand_coords = np.asarray(ligand_coords)

    n_lig = int(ligand_coords.shape[0])
    budget_atoms = max(0, int(max_atoms) - n_lig)
    budget_res = max(0, int(max_residues) - n_lig)

    # Group atoms by residue present in parsed pocket
    if protein_coords.size == 0:
        # Nothing to crop; return empty protein arrays and empty pocket
        return (
            protein_coords,
            protein_atom_types,
            protein_residue_types,
            protein_residue_indices,
        )

    uniq_res, inv = np.unique(protein_residue_indices, return_inverse=True)
    counts = np.bincount(inv, minlength=len(uniq_res))

    # Early exit if already within both budgets
    if int(protein_coords.shape[0]) <= budget_atoms and int(len(uniq_res)) <= budget_res:
        return protein_coords, protein_atom_types, protein_residue_types, protein_residue_indices

    # Distance-based ordering using all atoms in each residue
    if n_lig > 0:
        # Compute ligand center (mean of all ligand coordinates)
        ligand_center = np.mean(ligand_coords, axis=0)  # Shape: (3,)
        
        # Compute distance from each protein atom to ligand center
        diff = protein_coords - ligand_center[None, :]  # Shape: (n_protein_atoms, 3)
        atom_to_center_d2 = np.sum(diff * diff, axis=1)  # Shape: (n_protein_atoms,)
        
        # For each residue, find the minimum distance among all its atoms
        # Use np.minimum.reduceat for vectorized operation
        sort_idx = np.argsort(inv)
        sorted_atom_min_d2 = atom_to_center_d2[sort_idx]
        
        # Get split points for each residue
        split_points = np.cumsum(counts)[:-1]
        residue_min_d2 = np.minimum.reduceat(sorted_atom_min_d2, 
                                           np.concatenate(([0], split_points)))
        
        min_distances = np.sqrt(residue_min_d2)  # Calculate actual distances
        order = np.argsort(residue_min_d2)
        
        # Apply distance threshold filtering if specified
        if distance_threshold is not None:
            distance_mask = min_distances <= distance_threshold
            valid_residue_indices = np.where(distance_mask)[0]
            # Only keep residues within distance threshold
            filtered_order = order[np.isin(order, valid_residue_indices)]
            
            # If too few residues pass the distance threshold, keep the 100 closest
            if len(filtered_order) < 100:
                order = order[:100] if len(order) >= 100 else order
            else:
                order = filtered_order
    else:
        # Use natural residue order if no ligand atoms
        order = np.arange(len(uniq_res))

    counts_ordered = counts[order]
    csum = np.cumsum(counts_ordered)
    k_by_atoms = int(np.searchsorted(csum, budget_atoms, side='right'))
    k_by_res = min(int(budget_res), len(order))
    k = max(0, min(k_by_atoms, k_by_res))

    selected_pos = order[:k]
    selected_residues = uniq_res[selected_pos]

    # Atom mask for selected residues
    atom_mask = np.isin(protein_residue_indices, selected_residues)
    pc = protein_coords[atom_mask]
    pa = protein_atom_types[atom_mask]
    prt = protein_residue_types[atom_mask]
    pri = protein_residue_indices[atom_mask]

    return pc, pa, prt, pri

def build_edges(protein_coords: np.ndarray, ligand_coords: np.ndarray,
                protein_cutoff: float, cross_cutoff: float, 
                ligand_mol: Optional[Chem.Mol] = None, 
                protein_residue_indices: Optional[np.ndarray] = None,
                protein_atom_types: Optional[np.ndarray] = None,
                protein_residue_types: Optional[np.ndarray] = None,
                max_edges: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Build edges for the heterogeneous graph at atom level.
    
    Parameters
    ----------
    protein_coords : np.ndarray
        Protein atom coordinates [N_protein_atoms, 3]
    ligand_coords : np.ndarray
        Ligand coordinates [N_ligand, 3]
    protein_cutoff : float
        Distance cutoff for protein-protein edges
    cross_cutoff : float
        Distance cutoff for protein-ligand edges
    ligand_mol : Optional[Chem.Mol]
        Ligand molecule
    protein_residue_indices : Optional[np.ndarray]
        Protein residue indices for each atom [N_protein_atoms]
    protein_atom_types : Optional[np.ndarray]
        Protein atom types for each atom [N_protein_atoms]
    protein_residue_types : Optional[np.ndarray]
        Protein residue types for each atom [N_protein_atoms]
    max_edges : Optional[int]
        Maximum number of edges to keep. If exceeded, edges are sampled by priority
        
    Returns
    ------- 
    Tuple[np.ndarray, np.ndarray]
        edge_index [2, N_edges], edge_types [N_edges]
    """
    n_protein_atoms = len(protein_coords)
    n_ligand = len(ligand_coords)
    
    # Convert to torch tensors
    protein_coords_t = torch.tensor(protein_coords, dtype=torch.float32)
    ligand_coords_t = torch.tensor(ligand_coords, dtype=torch.float32)
    
    edge_list = []
    edge_type_list = []
    
    # 1. Protein-ligand edges (highest priority)
    if n_protein_atoms > 0 and n_ligand > 0:
        cross_dists = torch.cdist(protein_coords_t, ligand_coords_t)
        cross_mask = cross_dists <= cross_cutoff
        cross_edges = torch.where(cross_mask)
        
        if len(cross_edges[0]) > 0:
            protein_indices = cross_edges[0] + 1  # Offset by 1 for protein global node
            ligand_indices = cross_edges[1] + n_protein_atoms + 2  # Offset by 2 global nodes
            
            # Shuffle protein-ligand edges
            pl_edges = torch.stack([protein_indices, ligand_indices])
            pl_edge_types = torch.full((len(cross_edges[0]),), EDGE_TYPES['protein_ligand'])
            
            # Random permutation
            perm = torch.randperm(pl_edges.shape[1])
            pl_edges = pl_edges[:, perm]
            pl_edge_types = pl_edge_types[perm]

            edge_list.append(pl_edges)
            edge_type_list.append(pl_edge_types)
    
    # 2. Global edges (second priority)
    # P -> all protein atoms
    if n_protein_atoms > 0:
        p_to_protein_edges = torch.stack([
            torch.zeros(n_protein_atoms, dtype=torch.long),  # P node at index 0
            torch.arange(1, n_protein_atoms + 1, dtype=torch.long)  # protein atoms at indices 1..n_protein_atoms
        ])
        p_to_protein_types = torch.full((n_protein_atoms,), EDGE_TYPES['global'])
        edge_list.append(p_to_protein_edges)
        edge_type_list.append(p_to_protein_types)
    
    # L -> all ligand atoms
    if n_ligand > 0:
        l_to_ligand_edges = torch.stack([
            torch.full((n_ligand,), n_protein_atoms + 1, dtype=torch.long),  # L node at index n_protein_atoms + 1
            torch.arange(n_protein_atoms + 2, n_protein_atoms + n_ligand + 2, dtype=torch.long)  # ligand atoms at indices n_protein_atoms + 2..n_protein_atoms + n_ligand + 1
        ])
        l_to_ligand_types = torch.full((n_ligand,), EDGE_TYPES['global'])
        edge_list.append(l_to_ligand_edges)
        edge_type_list.append(l_to_ligand_types)
    
    # P <-> L bidirectional edges
    if n_protein_atoms > 0 and n_ligand > 0:
        p_to_l_edges = torch.tensor([[0], [n_protein_atoms + 1]], dtype=torch.long)  # P -> L
        l_to_p_edges = torch.tensor([[n_protein_atoms + 1], [0]], dtype=torch.long)  # L -> P
        p_l_edges = torch.cat([p_to_l_edges, l_to_p_edges], dim=1)
        p_l_types = torch.full((2,), EDGE_TYPES['global'])
        edge_list.append(p_l_edges)
        edge_type_list.append(p_l_types)
    
    # 3. Ligand internal edges - real bonds only (third priority)
    if ligand_mol is not None:
        edge_features = []  # [E, 3] - (node_in, node_out, edge_type)
        
        for bond in ligand_mol.GetBonds():
            start_atom_idx = bond.GetBeginAtomIdx()
            end_atom_idx = bond.GetEndAtomIdx()
            bond_type = bond.GetBondType()
            if bond_type not in LIGAND_BOND_TYPES:
                print(f"Warning: Bond type {bond_type} not found in LIGAND_BOND_TYPES")
                continue
            else:
                bond_id = LIGAND_BOND_TYPES[bond_type]
                edge_features.append([start_atom_idx, end_atom_idx, bond_id])
                edge_features.append([end_atom_idx, start_atom_idx, bond_id])

        edge_features = torch.tensor(edge_features, dtype=torch.long)

        # Real ligand bonds
        if len(edge_features) > 0:
            edge_features_t = torch.tensor(edge_features, dtype=torch.long)
            valid_mask = (edge_features_t[:, 0] >= 0) & (edge_features_t[:, 0] < n_ligand) & \
                        (edge_features_t[:, 1] >= 0) & (edge_features_t[:, 1] < n_ligand)
            valid_edges = edge_features_t[valid_mask]
            
            if len(valid_edges) > 0:
                ligand_edges = valid_edges[:, :2].T + n_protein_atoms + 2  # Offset by 2 global nodes
                ligand_edge_types = torch.tensor([EDGE_TYPES[f'ligand_bond_{et}'] for et in valid_edges[:, 2]])
                
                # Shuffle ligand bond edges
                perm = torch.randperm(ligand_edges.shape[1])
                ligand_edges = ligand_edges[:, perm]
                ligand_edge_types = ligand_edge_types[perm]
                
                edge_list.append(ligand_edges)
                edge_type_list.append(ligand_edge_types)
    
    # 4. Protein-protein sequence edges (fourth priority)
    if n_protein_atoms > 1 and protein_residue_indices is not None and protein_atom_types is not None:
        ca_atom_mask = protein_atom_types == 2  # CA is atom type 2
        
        if np.any(ca_atom_mask):
            ca_atom_indices = np.where(ca_atom_mask)[0]
            ca_residue_indices = protein_residue_indices[ca_atom_mask]
            
            if len(ca_residue_indices) > 1:
                ca_indices_t = torch.tensor(ca_atom_indices, dtype=torch.long)
                ca_res_indices_t = torch.tensor(ca_residue_indices, dtype=torch.long)
                
                i_indices, j_indices = torch.meshgrid(ca_indices_t, ca_indices_t, indexing='ij')
                i_res, j_res = torch.meshgrid(ca_res_indices_t, ca_res_indices_t, indexing='ij')
                
                res_diff = torch.abs(i_res - j_res)
                consecutive_mask = (res_diff == 1) & (i_indices != j_indices)
                
                seq_src = i_indices[consecutive_mask] + 1  # Offset by 1 for protein global node
                seq_dst = j_indices[consecutive_mask] + 1  # Offset by 1 for protein global node
                
                seq_edges = torch.stack([seq_src, seq_dst])
                seq_edge_types = torch.full((len(seq_src),), EDGE_TYPES['protein_sequence'])

                perm = torch.randperm(seq_edges.shape[1])
                seq_edges = seq_edges[:, perm]
                seq_edge_types = seq_edge_types[perm]

                edge_list.append(seq_edges)
                edge_type_list.append(seq_edge_types)
    
    # 5. Protein covalent edges (within residues) - fifth priority
    if n_protein_atoms > 1 and protein_residue_indices is not None and protein_atom_types is not None and protein_residue_types is not None:
        # Convert to tensors for vectorized operations
        protein_residue_indices_t = torch.tensor(protein_residue_indices, dtype=torch.long)
        protein_atom_types_t = torch.tensor(protein_atom_types, dtype=torch.long)
        protein_residue_types_t = torch.tensor(protein_residue_types, dtype=torch.long)
        
        # Find unique residues
        unique_residues = torch.unique(protein_residue_indices_t)
        
        covalent_edges_list = []
        
        # Process each residue (this loop is necessary but minimal)
        for residue_idx in unique_residues:
            # Find all atoms in this residue
            residue_mask = protein_residue_indices_t == residue_idx
            atoms_in_residue = torch.where(residue_mask)[0]
            
            if len(atoms_in_residue) < 2:
                continue
            
            # Create all possible atom pairs within this residue (vectorized)
            atom_pairs = torch.cartesian_prod(atoms_in_residue, atoms_in_residue)
            
            # Remove self-pairs
            valid_mask = atom_pairs[:, 0] != atom_pairs[:, 1]
            atom_pairs = atom_pairs[valid_mask]
            
            if len(atom_pairs) == 0:
                continue
            
            # Get atom types for the paired atoms
            atom1_types = protein_atom_types_t[atom_pairs[:, 0]]
            atom2_types = protein_atom_types_t[atom_pairs[:, 1]]
            
            # Get residue type
            residue_type = protein_residue_types_t[atoms_in_residue[0]]
            
            # Use global precomputed bond pattern table for ultra-fast lookup
            valid_bonds = BOND_PATTERN_TABLE[residue_type, atom1_types, atom2_types]
            
            if valid_bonds.any():
                # Add valid edges with +1 offset for global node
                valid_pairs = atom_pairs[valid_bonds] + 1
                covalent_edges_list.append(valid_pairs)
        
        if covalent_edges_list:
            # Combine all covalent edges
            covalent_edges = torch.cat(covalent_edges_list, dim=0).T
            covalent_edge_types = torch.full((covalent_edges.shape[1],), EDGE_TYPES['protein_covalent'])
            
            # Shuffle covalent edges
            perm = torch.randperm(covalent_edges.shape[1])
            covalent_edges = covalent_edges[:, perm]
            covalent_edge_types = covalent_edge_types[perm]
            
            edge_list.append(covalent_edges)
            edge_type_list.append(covalent_edge_types)
    
    # 6. Ligand LAS edges (sixth priority)
    if ligand_mol is not None:
        las_edge_index, _ = dense_to_sparse(get_LAS_distance_constraint_mask(ligand_mol))  # [2, K] - LAS edges
        valid_las_mask = (las_edge_index[0] >= 0) & (las_edge_index[0] < n_ligand) & \
                        (las_edge_index[1] >= 0) & (las_edge_index[1] < n_ligand)
        valid_las_edges = las_edge_index[:, valid_las_mask]
        if valid_las_edges.shape[1] > 0:
            las_edges = valid_las_edges + n_protein_atoms + 2  # Offset by 2 global nodes
            las_edge_types = torch.full((valid_las_edges.shape[1],), EDGE_TYPES['ligand_las'])
            
            # Shuffle LAS edges
            perm = torch.randperm(las_edges.shape[1])
            las_edges = las_edges[:, perm]
            las_edge_types = las_edge_types[perm]
            
            edge_list.append(las_edges)
            edge_type_list.append(las_edge_types)
    
    # 7. Protein-protein spatial edges (lowest priority) - only create edges between atoms from different residues
    if n_protein_atoms > 1 and protein_residue_indices is not None:
        dists = torch.cdist(protein_coords_t, protein_coords_t)
        spatial_mask = (dists > 0) & (dists <= protein_cutoff)
        spatial_edges = torch.where(spatial_mask)
        
        if len(spatial_edges[0]) > 0:
            # Filter out edges between atoms in the same residue
            protein_residue_indices_t = torch.tensor(protein_residue_indices, dtype=torch.long)
            residue_i = protein_residue_indices_t[spatial_edges[0]]
            residue_j = protein_residue_indices_t[spatial_edges[1]]
            
            # Only keep edges between atoms from different residues
            different_residue_mask = residue_i != residue_j
            filtered_edges = spatial_edges[0][different_residue_mask], spatial_edges[1][different_residue_mask]
            
            if len(filtered_edges[0]) > 0:
                spatial_edges = torch.stack(filtered_edges) + 1  # Offset by 1 for protein global node
                spatial_edge_types = torch.full((len(filtered_edges[0]),), EDGE_TYPES['protein_spatial'])

                perm = torch.randperm(spatial_edges.shape[1])
                spatial_edges = spatial_edges[:, perm]
                spatial_edge_types = spatial_edge_types[perm]

                edge_list.append(spatial_edges)
                edge_type_list.append(spatial_edge_types)
    
    # Concatenate all edges and deduplicate
    if edge_list:
        edge_index = torch.cat(edge_list, dim=1)
        edge_types = torch.cat(edge_type_list)
        
        # Convert to numpy for deduplication
        edge_index_np = edge_index.numpy().T  # [N_edges, 2]
        edge_types_np = edge_types.numpy()
        
        # Remove duplicate edges using numpy
        # Create a view of edges as structured array for easy comparison
        dtype = [('src', edge_index_np.dtype), ('dst', edge_index_np.dtype)]
        edges_structured = np.zeros(edge_index_np.shape[0], dtype=dtype)
        edges_structured['src'] = edge_index_np[:, 0]
        edges_structured['dst'] = edge_index_np[:, 1]
        
        # Find unique edges and their first occurrence indices
        _, unique_indices = np.unique(edges_structured, return_index=True)
        
        # Sort the unique indices to restore original appearance order (priority order)
        sorted_unique_indices = np.sort(unique_indices)
        
        # Get the unique edge indices and types in original priority order
        unique_edge_index = edge_index_np[sorted_unique_indices].T  # [2, N_unique_edges]
        unique_edge_types = edge_types_np[sorted_unique_indices]
        
        # Remove self-loops (edges where source == destination)
        no_self_loop_mask = unique_edge_index[0] != unique_edge_index[1]
        unique_edge_index = unique_edge_index[:, no_self_loop_mask]
        unique_edge_types = unique_edge_types[no_self_loop_mask]

        # Simple truncation if needed - edges are already in priority order
        if max_edges is not None and unique_edge_index.shape[1] > max_edges:
            unique_edge_index = unique_edge_index[:, :max_edges]
            unique_edge_types = unique_edge_types[:max_edges]
       
        sort_indices = np.lexsort((unique_edge_index[1], unique_edge_index[0]))

        unique_edge_index = unique_edge_index[:, sort_indices]
        unique_edge_types = unique_edge_types[sort_indices]

        return unique_edge_index, unique_edge_types
    else:
        # Return empty edges
        return np.array([[0], [0]], dtype=np.int64), np.array([0], dtype=np.int64)