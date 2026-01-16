"""
Affinity Inference Dataset

This module contains the dataset and data module for affinity inference,
with simplified data loading directly from PDB and SDF files.
"""

from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass
from rdkit.Chem import Descriptors
import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader
import random
from affinity.utils import extract_ids
from affinity.utils.resource_loader import ResourceLoader
from affinity.data.repr.torchdrug import extract_feature
from .featurizer import (
    build_edges,
    crop_pocket
)
from .parser import (
    parse_structure_file,
    parse_sdf_file,
)
from .training import load_id_list


@dataclass
class AffinityInferenceDataset:
    """Simplified affinity inference dataset."""
    
    structure_loader: ResourceLoader
    ligand_loader: ResourceLoader
    protein_repr_loader: ResourceLoader
    id_list: List[str]
    ligand_repr_loader: Optional[ResourceLoader] = None
    morgan_repr_loader: Optional[ResourceLoader] = None
    unimol_repr_loader: Optional[ResourceLoader] = None
    pocket_indices_loader: Optional[ResourceLoader] = None
    max_atoms: int = 2048
    max_residues: int = 512
    max_edges: int = 16384
    protein_edge_cutoff: float = 4.0
    cross_edge_cutoff: float = 10.0
    distance_threshold: float = 20.0

    @classmethod
    def from_config(
        cls,
        structure: str,
        structure_type: str,
        ligand: str,
        ligand_type: str,
        protein_repr: str,
        ligand_repr: str,
        id_list: str,
        morgan_repr: Optional[str] = None,
        unimol_repr: Optional[str] = None,
        pocket_indices: Optional[str] = None,
        max_atoms: int = 2048,
        max_residues: int = 512,
        max_edges: int = 16384,
        protein_edge_cutoff: float = 4.0,
        cross_edge_cutoff: float = 10.0,
        distance_threshold: float = 20.0,
    ) -> "AffinityInferenceDataset":

        structure_loader = ResourceLoader(Path(structure), extension="." + structure_type)
        ligand_loader = ResourceLoader(Path(ligand), extension="." + ligand_type)
        pocket_indices_loader = ResourceLoader(Path(pocket_indices)) if pocket_indices else None
        protein_repr_loader = ResourceLoader(Path(protein_repr))
        ligand_repr_loader = ResourceLoader(Path(ligand_repr)) if ligand_repr else None
        morgan_repr_loader = ResourceLoader(Path(morgan_repr)) if morgan_repr else None
        unimol_repr_loader = ResourceLoader(Path(unimol_repr)) if unimol_repr else None
        id_list = load_id_list(id_list)
        return cls(
            structure_loader=structure_loader,
            ligand_loader=ligand_loader,
            pocket_indices_loader=pocket_indices_loader,
            protein_repr_loader=protein_repr_loader,
            ligand_repr_loader=ligand_repr_loader,
            morgan_repr_loader=morgan_repr_loader,
            unimol_repr_loader=unimol_repr_loader,
            id_list=id_list,
            max_atoms=max_atoms,
            max_residues=max_residues,
            max_edges=max_edges,
            protein_edge_cutoff=protein_edge_cutoff,
            cross_edge_cutoff=cross_edge_cutoff,
            distance_threshold=distance_threshold,
        )


def collate_affinity_inference(data: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Collate the data for affinity inference.
    
    For inference, we don't need padding since we process samples one by one.
    Adds a batch dimension to tensors expected by the model (batch=1).
    
    Parameters
    ----------
    data : List[Dict[str, Tensor]]
        The data to collate.
        
    Returns
    -------
    Dict[str, Tensor]
        The collated data with batch dimension.
    """
    if not data or data[0] is None:
        return {}
    sample = data[0]
    out: dict[str, Tensor] = {}

    # Keys that require a batch dimension [1, ...]
    keys_to_unsqueeze = [
        'coords', 'atom_types', 'molecule_types', 'residue_types', 'residue_indices',
        'protein_repr', 'ligand_repr', 'atom_mask', 'edge_mask', 'edge_index', 'edge_types', 'mw', 'morgan_repr', 'unimol_repr'
    ]

    for k, v in sample.items():
        if isinstance(v, torch.Tensor) and k in keys_to_unsqueeze:
            out[k] = v.unsqueeze(0)
        else:
            out[k] = v

    return out


class AffinityInferenceDatasets(torch.utils.data.Dataset):
    """Simplified affinity inference dataset."""

    def __init__(
        self,
        datasets: List[AffinityInferenceDataset],
    ) -> None:
        """Initialize the dataset.

        Parameters
        ----------
        datasets : List[AffinityInferenceDataset]
            List of datasets to sample from.
        """
        super().__init__()
        self.datasets = datasets

        # Collect all IDs
        self.all_ids = []
        self.dataset_indices = []

        for i, dataset in enumerate(datasets):
            self.all_ids.extend(dataset.id_list)
            self.dataset_indices.extend([i] * len(dataset.id_list))

    def __getitem__(self, idx: int) -> dict:
        """Get a single inference sample.

        Parameters
        ----------
        idx : int
            The index of the item.

        Returns
        -------
        Dict[str, Tensor]
            The sampled data features.
        """
        sample_id = self.all_ids[idx]
        dataset_idx = self.dataset_indices[idx]
        dataset = self.datasets[dataset_idx]

        # Load and process the sample
        try:
            return self._process_sample(sample_id, dataset)
        except Exception as e:
            print(f"Error processing {sample_id}: {e}")
            return None

    def _process_sample(self, sample_id: str, dataset: AffinityInferenceDataset) -> dict:
        """Process a single sample into features.

        Parameters
        ----------
        sample_id : str
            The sample ID (format: {prot_id}_{ligand_id}).
        dataset : AffinityInferenceDataset
            The dataset containing the sample.

        Returns
        -------
        dict
            The processed features.
        """
        prot_id, ligand_id = extract_ids(sample_id)
        
        pocket_indices = dataset.pocket_indices_loader.get(sample_id) if dataset.pocket_indices_loader is not None else None

        if dataset.ligand_loader.extension == '.sdf':
            structure_content = dataset.structure_loader.get(prot_id)
            if isinstance(structure_content, list): # for multiple structures in unique protein
                structure_content = random.choice(structure_content)
            # Protein mode: structure contains the entire protein
            protein_coords, protein_atom_types, protein_residue_types, protein_residue_indices = parse_structure_file(
                structure_content, pocket_indices, dataset.structure_loader.extension[1:], "protein"
            )
            # Load ligand from SDF
            sdf_content = dataset.ligand_loader.get(sample_id)
            if isinstance(sdf_content, list): # for multiple structures in unique ligand
                sdf_content = random.choice(sdf_content)
            ligand_coords, ligand_atom_types, ligand_mol = parse_sdf_file(sdf_content)
        else:
            # Complex mode: structure contains the entire complex
            structure_content = dataset.structure_loader.get(sample_id)
            if isinstance(structure_content, list): # for multiple structures in unique complex
                structure_content = random.choice(structure_content)
            smiles = dataset.ligand_loader.get(ligand_id)
            protein_coords, protein_atom_types, protein_residue_types, protein_residue_indices, \
            ligand_coords, ligand_atom_types, ligand_mol = parse_structure_file(
                structure_content, pocket_indices, dataset.structure_loader.extension[1:], "complex", smiles
            )

        assert protein_coords.shape[0] > 0, f"protein_coords.shape[0] {protein_coords.shape[0]} == 0"
        assert ligand_coords.shape[0] > 0, f"ligand_coords.shape[0] {ligand_coords.shape[0]} == 0"

        protein_repr = dataset.protein_repr_loader.get(prot_id)
        if dataset.ligand_repr_loader is not None:
            ligand_repr = dataset.ligand_repr_loader.get(ligand_id)
        else:
            ligand_repr = extract_feature(ligand_mol)
        
        # Load Morgan representation (optional)
        morgan_repr = None
        if dataset.morgan_repr_loader is not None:
            morgan_repr = dataset.morgan_repr_loader.get(ligand_id)
        
        # Load UniMol representation (optional)
        unimol_repr = None
        if dataset.unimol_repr_loader is not None:
            unimol_repr = dataset.unimol_repr_loader.get(ligand_id)

        # Optional cropping based on budgets (do not error at inference)
        n_protein_atoms = len(protein_coords)
        n_ligand_atoms = len(ligand_coords)
        unique_residue_indices = np.unique(protein_residue_indices)
        total_residue_count = len(unique_residue_indices) + n_ligand_atoms
        if (n_protein_atoms + n_ligand_atoms) > dataset.max_atoms or total_residue_count > dataset.max_residues:
            (
                protein_coords,
                protein_atom_types,
                protein_residue_types,
                protein_residue_indices,
            ) = crop_pocket(
                protein_coords=protein_coords,
                protein_atom_types=protein_atom_types,
                protein_residue_types=protein_residue_types,
                protein_residue_indices=protein_residue_indices,
                ligand_coords=ligand_coords,
                max_atoms=dataset.max_atoms,
                max_residues=dataset.max_residues,
                distance_threshold=dataset.distance_threshold,
            )

        unique_residue_indices = np.unique(protein_residue_indices)
        # Extract pocket part from protein_repr and expand to atom-level
        # Assuming protein_repr is [N_residues, feature_dim]
        if protein_repr is not None and len(protein_coords) > 0:
            protein_repr_np = np.array(protein_repr)
            
            # Get unique residue indices from the actual protein atoms
            pocket_protein_repr = protein_repr_np[unique_residue_indices]  # [n_pocket_residues, feature_dim]
            
            # Create mapping from residue index to pocket index using vectorized operations
            residue_to_pocket_idx = np.full(protein_repr_np.shape[0], -1, dtype=int)
            residue_to_pocket_idx[unique_residue_indices] = np.arange(len(unique_residue_indices))
            
            # Get pocket indices for each atom's residue using vectorized indexing
            atom_pocket_indices = residue_to_pocket_idx[protein_residue_indices]
            
            # Extract repr for atoms (all atoms should be valid now)
            protein_repr = pocket_protein_repr[atom_pocket_indices]

        assert protein_repr.shape[0] == protein_coords.shape[0], f"protein_repr.shape[0] {protein_repr.shape[0]} != protein_coords.shape[0] {protein_coords.shape[0]}"
        assert ligand_repr.shape[0] == ligand_coords.shape[0], f"ligand_repr.shape[0] {ligand_repr.shape[0]} != ligand_coords.shape[0] {ligand_coords.shape[0]}"

        # Recompute sizes
        n_protein_atoms = len(protein_coords)
        n_ligand_atoms = len(ligand_coords)
        n_total_atoms = n_protein_atoms + n_ligand_atoms

        # Build edges (align args with training; no edge subsample at inference)
        edge_index, edge_types = build_edges(
            protein_coords,
            ligand_coords,
            dataset.protein_edge_cutoff,
            dataset.cross_edge_cutoff,
            ligand_mol,
            protein_residue_indices,
            protein_atom_types,
            protein_residue_types,
            dataset.max_edges,
        )

        # Global nodes
        if n_protein_atoms > 0:
            p_coords = protein_coords.mean(axis=0)
            if protein_repr is not None and len(protein_repr) > 0:
                p_global_repr = protein_repr.mean(axis=0)
            else:
                p_global_repr = np.zeros(384)
        else:
            p_coords = np.zeros(3)
            p_global_repr = np.zeros(384)

        if n_ligand_atoms > 0:
            l_coords = ligand_coords.mean(axis=0)
            if ligand_repr is not None and len(ligand_repr) > 0:
                l_global_repr = ligand_repr.mean(axis=0, dtype=torch.float32)
            else:
                l_global_repr = np.zeros(56)
        else:
            l_coords = np.zeros(3)
            l_global_repr = np.zeros(56)

        
        # Final coordinate order: [P, protein_atoms, L, ligand_atoms]
        all_coords = np.concatenate([
            p_coords.reshape(1, 3),
            protein_coords,
            l_coords.reshape(1, 3),
            ligand_coords
        ], axis=0)
        
        # Atom/molecule/residue types
        all_atom_types = np.concatenate([
            np.array([-1]),
            protein_atom_types,
            np.array([-1]),
            ligand_atom_types
        ], axis=0)
        
        molecule_types = np.concatenate([
            np.array([1]),
            np.ones(n_protein_atoms, dtype=np.int64),
            np.array([2]),
            np.full(n_ligand_atoms, 2, dtype=np.int64)
        ], axis=0)
        
        all_residue_types = np.concatenate([
            np.array([-1]),
            protein_residue_types,
            np.array([-1]),
            ligand_atom_types
        ], axis=0)
        
        all_residue_indices = np.concatenate([
            np.array([-1]),
            protein_residue_indices,
            np.array([-1]),
            np.arange(n_ligand_atoms) + (protein_residue_indices.max() + 1 if len(protein_residue_indices) > 0 else 0)
        ], axis=0)
        
        # Combine representations with global nodes
        combined_protein_repr = np.concatenate([
            p_global_repr.reshape(1, -1),
            protein_repr if protein_repr is not None else np.zeros((n_protein_atoms, 384))
        ], axis=0)
        
        combined_ligand_repr = np.concatenate([
            l_global_repr.reshape(1, -1),
            ligand_repr if ligand_repr is not None else np.zeros((n_ligand_atoms, 56))
        ], axis=0)
        
        # Masks
        atom_mask = np.ones(n_total_atoms + 2, dtype=bool)
        edge_mask = np.ones(edge_index.shape[1], dtype=bool)
        
        mw = Descriptors.MolWt(ligand_mol)

        # To tensors
        features = {
            'coords': torch.tensor(all_coords, dtype=torch.float32),
            'atom_types': torch.tensor(all_atom_types, dtype=torch.long),
            'molecule_types': torch.tensor(molecule_types, dtype=torch.long),
            'residue_types': torch.tensor(all_residue_types, dtype=torch.long),
            'residue_indices': torch.tensor(all_residue_indices, dtype=torch.long),
            'protein_repr': torch.tensor(combined_protein_repr, dtype=torch.float32),
            'ligand_repr': torch.tensor(combined_ligand_repr, dtype=torch.float32),
            'atom_mask': torch.tensor(atom_mask, dtype=torch.bool),
            'edge_mask': torch.tensor(edge_mask, dtype=torch.bool),
            'edge_index': torch.tensor(edge_index, dtype=torch.long),
            'edge_types': torch.tensor(edge_types, dtype=torch.long),
            'mw': torch.tensor(mw, dtype=torch.float32),
        }
        
        # Add Morgan representation if available
        if morgan_repr is not None:
            features['morgan_repr'] = torch.tensor(morgan_repr, dtype=torch.float32)
        
        # Add UniMol representation if available
        if unimol_repr is not None:
            features['unimol_repr'] = torch.tensor(unimol_repr, dtype=torch.float32)
        
        # Record ID
        id_str = sample_id + '-' * (40 - len(sample_id))
        id_tensor = torch.tensor([ord(c) for c in id_str], dtype=torch.int32)
        features['record_id'] = id_tensor
        
        return features

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns
        -------
        int
            The length of the dataset.
        """
        return len(self.all_ids)


class AffinityInferenceDataModule(pl.LightningDataModule):
    """Simplified DataModule for affinity inference."""

    def __init__(
        self,
        datasets: List[AffinityInferenceDataset],
        num_workers: int = 0,
        batch_size: int = 1,
    ) -> None:
        """Initialize the DataModule.

        Parameters
        ----------
        datasets : List[AffinityInferenceDataset]
            Datasets to use for inference.
        num_workers : int
            The number of workers to use.
        batch_size : int
            Batch size for inference.
        """
        super().__init__()
        self.datasets = datasets
        self.num_workers = num_workers
        self.batch_size = batch_size
        assert self.batch_size == 1, f"batch size should be 1 for inference"

    def predict_dataloader(self) -> DataLoader:
        """Get the prediction dataloader.

        Returns
        -------
        DataLoader
            The prediction dataloader.
        """
        dataset = AffinityInferenceDatasets(
            datasets=self.datasets,
        )
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            collate_fn=collate_affinity_inference,
        )

    def transfer_batch_to_device(
        self,
        batch: dict,
        device: torch.device,
        dataloader_idx: int,  # noqa: ARG002
    ) -> dict:
        """Transfer a batch to the given device.

        Parameters
        ----------
        batch : Dict
            The batch to transfer.
        device : torch.device
            The device to transfer to.
        dataloader_idx : int
            The dataloader index.

        Returns
        -------
        Dict
            The transferred batch.
        """
        for key in batch:
            if key in ['coords', 'atom_types', 'molecule_types', 'residue_types', 'residue_indices',
                    'protein_repr', 'ligand_repr', 'atom_mask', 'protein_mask', 'ligand_mask',
                    'edge_index', 'edge_types', 'record_id']:
                batch[key] = batch[key].to(device)
        return batch