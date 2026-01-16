"""
Affinity Training Dataset

This module contains the dataset and data module for affinity training,
with simplified data loading directly from PDB and SDF files.
"""
    
import json
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass
import random
import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader, DistributedSampler

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
import torch.distributed as dist
from .sampler import BalancedGroupBatchSampler

@dataclass
class DatasetResources:
    """Actual dataset with loaded resources."""
    
    type: str
    structure_loader: ResourceLoader
    protein_repr_loader: ResourceLoader
    label_loader: ResourceLoader
    id_list: List[str]
    ligand_loader: ResourceLoader
    ligand_repr_loader: Optional[ResourceLoader] = None
    morgan_repr_loader: Optional[ResourceLoader] = None
    unimol_repr_loader: Optional[ResourceLoader] = None
    pocket_indices_loader: Optional[ResourceLoader] = None
    prob: Optional[float] = None
    overfit: Optional[int] = None  # Only for debug

@dataclass
class AffinityDataConfig:
    """Data module config."""
    
    train_sets: dict
    val_sets: Optional[dict] = None
    max_atoms: int = 2048
    max_edges: int = 16384
    max_residues: int = 512
    protein_edge_cutoff: float = 4.0
    cross_edge_cutoff: float = 10.0
    distance_threshold: Optional[float] = None

def load_id_list(id_list: str) -> List[str]:
    """Load list of sample IDs from various sources.
    
    Parameters
    ----------
    id_list : str
        Path to file containing ID list (JSON, TXT, etc.)
        
    Returns
    -------
    List[str]
        List of sample IDs
    """
    source_path = Path(id_list)
    
    if source_path.suffix == '.json':
        with source_path.open('r') as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                return list(data.keys())
            else:
                raise ValueError(f"JSON data must be list or dict, got {type(data)}")
    elif source_path.suffix == '.txt':
        with source_path.open('r') as f:
            return [line.strip() for line in f if line.strip()]
    else:
        raise ValueError(f"Unsupported ID list format: {source_path.suffix}")


def process_affinity_sample(sample_id: str, dataset: DatasetResources, dataset_idx: int,
                           global_cfg: AffinityDataConfig) -> dict:
    """Process a single sample into features.
    
    Parameters
    ----------
    sample_id : str
        The sample ID (format: {prot_id}_{ligand_id}).
    dataset : DatasetResources
        The dataset containing the sample.
    dataset_idx : int
        The index of the dataset.
    global_cfg : AffinityDataConfig
        Global data configuration.
        
    Returns
    -------
    dict
        The processed features.
    """
    prot_id, ligand_id = extract_ids(sample_id)
    # Load pocket indices (optional)
    if dataset.pocket_indices_loader is not None:
        pocket_indices = dataset.pocket_indices_loader.get(sample_id)
    else:
        pocket_indices = None

    # Determine parsing mode based on ligand_loader availability
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

    # Load representations
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

    # If exceeds budgets, crop pocket by CA-to-ligand distance ordering
    n_protein_atoms = len(protein_coords)
    n_ligand_atoms = len(ligand_coords)
    unique_residue_indices = np.unique(protein_residue_indices)
    total_residue_count = len(unique_residue_indices) + n_ligand_atoms
    if (n_protein_atoms + n_ligand_atoms) > global_cfg.max_atoms or total_residue_count > global_cfg.max_residues:
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
            max_atoms=global_cfg.max_atoms,
            max_residues=global_cfg.max_residues,
            distance_threshold=global_cfg.distance_threshold,
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
    
    # For atom-level processing, we use all atoms
    n_protein_atoms = len(protein_coords)
    n_ligand_atoms = len(ligand_coords)

    n_total_atoms = n_protein_atoms + n_ligand_atoms  
    
    # Check limits
    if n_total_atoms > global_cfg.max_atoms:
        raise ValueError(f"Total atoms {n_total_atoms} exceeds max_atoms {global_cfg.max_atoms}")
    
    # Check residue limit (pocket residues + ligand atoms)
    total_residue_count = len(unique_residue_indices) + n_ligand_atoms
    if total_residue_count > global_cfg.max_residues:
        raise ValueError(f"Total residue count {total_residue_count} (pocket: {len(unique_residue_indices)} + ligand: {n_ligand_atoms}) exceeds max_residues {global_cfg.max_residues}")
    
    # Build edges
    edge_index, edge_types = build_edges(
        protein_coords, ligand_coords,
        global_cfg.protein_edge_cutoff,
        global_cfg.cross_edge_cutoff,
        ligand_mol,
        protein_residue_indices,  # Pass atom residue indices for sequence edge construction
        protein_atom_types,  # Pass atom types for CA atom identification
        protein_residue_types,  # Pass residue types for covalent bond construction
        global_cfg.max_edges,  # Pass max_edges for priority sampling
    )

    # Create global nodes first
    # Create protein global node (P)
    if n_protein_atoms > 0:
        p_coords = protein_coords.mean(axis=0)
        if protein_repr is not None and len(protein_repr) > 0:
            p_global_repr = protein_repr.mean(axis=0)
        else:
            p_global_repr = np.zeros(384)  # Default protein repr dim
    else:
        p_coords = np.zeros(3)
        p_global_repr = np.zeros(384)
    
    # Create ligand global node (L)
    if n_ligand_atoms > 0:
        l_coords = ligand_coords.mean(axis=0)
        if ligand_repr is not None and len(ligand_repr) > 0:
            l_global_repr = ligand_repr.mean(axis=0, dtype=torch.float32)
        else:
            l_global_repr = np.zeros(56)  # Default ligand repr dim
    else:
        l_coords = np.zeros(3)
        l_global_repr = np.zeros(56)
    
    # Final coordinate order: [P, protein_atoms, L, ligand_atoms]
    all_coords = np.concatenate([
        p_coords.reshape(1, 3),  # Protein global node
        protein_coords,          # Protein atoms
        l_coords.reshape(1, 3),  # Ligand global node
        ligand_coords           # Ligand atoms
    ], axis=0)
    
    # Create atom types and molecule types with global nodes
    # Global nodes use -1 as special flag (global node)
    all_atom_types = np.concatenate([
        np.array([-1]),  # Protein global node (-1 = global node)
        protein_atom_types,
        np.array([-1]),  # Ligand global node (-1 = global node)
        ligand_atom_types
    ], axis=0)
    
    molecule_types = np.concatenate([
        np.array([1]),  # Protein global node (1 for protein)
        np.ones(n_protein_atoms, dtype=np.int64),  # Protein atoms
        np.array([2]),  # Ligand global node (2 for ligand)
        np.full(n_ligand_atoms, 2, dtype=np.int64)     # Ligand atoms
    ], axis=0)
    
    # Create residue types with global nodes
    all_residue_types = np.concatenate([
        np.array([-1]),  # Protein global node (-1 = global node)
        protein_residue_types, 
        np.array([-1]),  # Ligand global node (-1 = global node)
        ligand_atom_types 
    ], axis=0)
    
    # Create residue indices with global nodes (-1 = global node, 0 = padding)
    all_residue_indices = np.concatenate([
        np.array([-1]),  # Protein global node (-1 = global node)
        protein_residue_indices, 
        np.array([-1]),  # Ligand global node (-1 = global node)
        np.arange(n_ligand_atoms)  + (protein_residue_indices.max() + 1 if len(protein_residue_indices) > 0 else 0)
    ], axis=0)
    
    # Load label based on dataset type
    label_data = dataset.label_loader.get(sample_id)
    
    if isinstance(label_data["label"], list):
        label_data["label"] = label_data["label"][0]

    if dataset.type == "binary" or dataset.type == "enzyme":
        label = 1 if label_data["label"] == "True" else 0
    elif dataset.type == "value":
        label = float(label_data["label"])
    else:
        raise ValueError(f"Unknown dataset type: {dataset.type}")
    
    # Combine protein and ligand representations with global nodes
    combined_protein_repr = np.concatenate([
        p_global_repr.reshape(1, -1),  # Protein global node repr
        protein_repr if protein_repr is not None else np.zeros((n_protein_atoms, 384))
    ], axis=0)
    
    combined_ligand_repr = np.concatenate([
        l_global_repr.reshape(1, -1),  # Ligand global node repr
        ligand_repr if ligand_repr is not None else np.zeros((n_ligand_atoms, 56))
    ], axis=0)
    
    # Create masks including global nodes
    atom_mask = np.ones(n_total_atoms + 2, dtype=bool)
    edge_mask = np.ones(edge_index.shape[1], dtype=bool)

    # Convert to tensors
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
        'dataset_idx': torch.tensor(dataset_idx, dtype=torch.long),
    }
    
    # Add Morgan representation if available
    if morgan_repr is not None:
        features['morgan_repr'] = torch.tensor(morgan_repr, dtype=torch.float32)
    
    # Add UniMol representation if available
    if unimol_repr is not None:
        features['unimol_repr'] = torch.tensor(unimol_repr, dtype=torch.float32)
    
    # Add unified label and label type
    if label is not None:
        features['label'] = torch.tensor(label, dtype=torch.float32)
        if dataset.type == "binary":
            features['label_type'] = torch.tensor(1, dtype=torch.long)  # 1 for binary
        elif dataset.type == "value":
            features['label_type'] = torch.tensor(2, dtype=torch.long)  # 2 for value
        elif dataset.type == "enzyme":
            features['label_type'] = torch.tensor(3, dtype=torch.long)  # 3 for enzyme
        else:
            raise ValueError(f"Unknown dataset type: {dataset.type}")
    
    # Add record ID
    id_str = sample_id + '-' * (100 - len(sample_id))
    id_tensor = torch.tensor([ord(c) for c in id_str], dtype=torch.int32)
    features['record_id'] = id_tensor
    
    return features

def collate_affinity(data: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Collate the data for affinity training.

    Parameters
    ----------
    data : List[Dict[str, Tensor]]
        The data to collate.

    Returns
    -------
    Dict[str, Tensor]
        The collated data.
    """
    # Record original batch size for DDP consistency
    expected_batch_size = len(data)
    
    # Filter out None samples
    valid_data = [d for d in data if d is not None]
    if not valid_data:
        return {}
    
    # Pad to expected batch size by duplicating valid samples
    # This ensures all ranks have the same batch size for DDP synchronization
    # Duplicates will be removed by deduplication logic in on_validation_epoch_end
    num_valid = len(valid_data)
    while len(valid_data) < expected_batch_size:
        idx = (len(valid_data) - num_valid) % num_valid
        valid_data.append(valid_data[idx])
    
    # Get the keys
    keys = valid_data[0].keys()

    # Collate the data
    collated = {}
    for key in keys:
        values = [d[key] for d in valid_data if key in d]
        if not values:
            continue
        
        # Handle different types of data
        if key in ['coords', 'atom_types', 'molecule_types', 'residue_types', 'residue_indices',
                'protein_repr', 'ligand_repr', 'atom_mask', 'edge_types', 'edge_mask']:
            # Node features need padding - pad to batch max size
            values = torch.nn.utils.rnn.pad_sequence(values, batch_first=True, padding_value=0)
        elif key in ['edge_index']:
            # edge_index is [2, E] - transpose to [E, 2], pad, then transpose back
            values = [v.transpose(0, 1) for v in values]  # [E, 2]
            values = torch.nn.utils.rnn.pad_sequence(values, batch_first=True, padding_value=0)  # [batch_size, max_edges, 2]
            values = values.transpose(1, 2)  # [batch_size, 2, max_edges]
        elif key in ['label', 'label_type', 'dataset_idx', 'record_id', 'morgan_repr', 'unimol_repr']:
            values = torch.stack(values, dim=0)
        else:
            # Default stacking
            print(f"Stacking {key} with default stacking")
            try:
                values = torch.stack(values, dim=0)
            except:
                # If shapes differ, use padding
                values = torch.nn.utils.rnn.pad_sequence(values, batch_first=True, padding_value=ord('-'))

        collated[key] = values

    return collated

class GroupedCollate:
    """Callable collate that drops any incomplete protein-group (size = group_size).
    The order of samples is preserved; groups are contiguous in the batch as produced by the batch sampler.
    If all groups are dropped, returns an empty dict so that training can safely skip the step in DDP.
    """

    def __init__(self, group_size: int) -> None:
        self.group_size = int(group_size)

    def __call__(self, data: list[Optional[dict]]) -> dict[str, Tensor]:
        if not data:
            return {}
        # Split into contiguous groups of size group_size
        kept: list[dict] = []
        n = len(data)
        # If length is not divisible, we drop the tail to avoid mixing groups
        usable = (n // self.group_size) * self.group_size
        for start in range(0, usable, self.group_size):
            group = data[start:start + self.group_size]
            if any(item is None for item in group):
                continue
            kept.extend(group)
        if not kept:
            return {}
        # Reuse the original collate for actual tensor padding/stacking
        return collate_affinity(kept)  # type: ignore[arg-type]

class AffinityTrainingDatasets(torch.utils.data.Dataset):
    """Simplified affinity training dataset."""

    def __init__(
        self,
        datasets: list[DatasetResources],
        global_cfg: AffinityDataConfig,
    ) -> None:
        """Initialize the dataset.

        Parameters
        ----------
        datasets : list[DatasetResources]
            List of datasets to sample from.
        global_cfg : AffinityDataConfig
            Global data configuration.
        """
        super().__init__()
        self.datasets = datasets
        self.global_cfg = global_cfg

        # Collect all IDs
        self.all_ids = []
        self.dataset_indices = []

        for i, dataset in enumerate(datasets):
            if dataset.overfit is not None:
                ids = dataset.id_list[:dataset.overfit]
                print(f"Overfitting to {len(ids)} records")
            else:
                ids = dataset.id_list
            
            self.all_ids.extend(ids)
            self.dataset_indices.extend([i] * len(ids))

    def __getitem__(self, idx: int) -> Optional[dict]:
        """Get a single training sample.

        Parameters
        ----------
        idx : int
            The index of the item.g

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
            return process_affinity_sample(sample_id, dataset, dataset_idx, self.global_cfg)
        except Exception as e:
            print(f"Error processing {sample_id} from dataset {dataset_idx}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns
        -------
        int
            The length of the dataset.
        """
        return len(self.all_ids)


class AffinityValidationDatasets(torch.utils.data.Dataset):
    """Simplified affinity validation dataset."""

    def __init__(
        self,
        datasets: list[DatasetResources],
        global_cfg: AffinityDataConfig,
    ) -> None:
        """Initialize the dataset.

        Parameters
        ----------
        datasets : list[DatasetResources]
            List of datasets to sample from.
        global_cfg : AffinityDataConfig
            Global data configuration.
        """
        super().__init__()
        self.datasets = datasets
        self.global_cfg = global_cfg
        self.all_ids = []
        self.dataset_indices = []
        
        for i, dataset in enumerate(datasets):
            if dataset.overfit is not None:
                ids = dataset.id_list[:dataset.overfit]
            else:
                ids = dataset.id_list
            
            self.all_ids.extend(ids)
            self.dataset_indices.extend([i] * len(ids))

    def __len__(self):
        return len(self.all_ids)

    def __getitem__(self, idx: int) -> Optional[dict]:
        """Get a single validation sample.

        Parameters
        ----------
        idx : int
            The index of the item.
        """
        sample_id = self.all_ids[idx]
        dataset_idx = self.dataset_indices[idx]
        dataset = self.datasets[dataset_idx]

        # Load and process the sample
        try:
            return process_affinity_sample(sample_id, dataset, dataset_idx, self.global_cfg)
        except Exception as e:
            print(f"Error processing {sample_id} from dataset {dataset_idx}: {e}")
            import traceback
            traceback.print_exc()
            return None

class AffinityTrainingDataModule(pl.LightningDataModule):
    """Simplified DataModule for affinity training."""

    def __init__(self, cfg: AffinityDataConfig) -> None:
        """Initialize the DataModule.

        Parameters
        ----------
        cfg : AffinityDataConfig
            The data configuration.
        """
        super().__init__()
        self.cfg = cfg

        # Load train and validation datasets using unified logic
        self.train_datasets = self._load_datasets_from_configs(
            cfg.train_sets.datasets, "train"
        )
        
        if not self.train_datasets:
            raise ValueError("No valid train datasets found")

        # Load validation datasets
        self.val_datasets = []
        if cfg.val_sets:
            self.val_datasets = self._load_datasets_from_configs(
                cfg.val_sets.datasets, "validation"
            )
            if not self.val_datasets:
                print("Warning: No validation datasets found")
        else:
            print("Warning: No validation datasets found")

    def _create_dataset_from_config(self, dataset_config, dataset_type: str = "train") -> DatasetResources:
        """Create a single dataset from configuration.
        
        Parameters
        ----------
        dataset_config : object
            Dataset configuration object
        dataset_type : str
            Type of dataset ("train" or "validation")
            
        Returns
        -------
        DatasetResources
            The created dataset
        """
        print(f"Loading {dataset_type} dataset from: {dataset_config.label}")
        
        # Create resource loaders
        structure_loader = ResourceLoader(Path(dataset_config.structure), extension="." + dataset_config.structure_type)
        ligand_loader = ResourceLoader(Path(dataset_config.ligand), extension="." + dataset_config.ligand_type)
        pocket_indices_loader = None
        if hasattr(dataset_config, 'pocket_indices') and getattr(dataset_config, 'pocket_indices'):
            pocket_indices_loader = ResourceLoader(Path(dataset_config.pocket_indices))

        ligand_repr_loader = None
        if hasattr(dataset_config, 'ligand_repr') and getattr(dataset_config, 'ligand_repr'):
            ligand_repr_loader = ResourceLoader(Path(dataset_config.ligand_repr))

        morgan_repr_loader = None
        if hasattr(dataset_config, 'morgan_repr') and getattr(dataset_config, 'morgan_repr'):
            morgan_repr_loader = ResourceLoader(Path(dataset_config.morgan_repr))

        unimol_repr_loader = None
        if hasattr(dataset_config, 'unimol_repr') and getattr(dataset_config, 'unimol_repr'):
            unimol_repr_loader = ResourceLoader(Path(dataset_config.unimol_repr))

        protein_repr_loader = ResourceLoader(Path(dataset_config.protein_repr))
        label_loader = ResourceLoader(Path(dataset_config.label))
        
        # Load ID list
        id_list = load_id_list(dataset_config.id_list)
        print(f"Loaded {len(id_list)} {dataset_type} samples")
        
        # Build base dataset kwargs
        dataset_kwargs = {
            'structure_loader': structure_loader,
            'ligand_loader': ligand_loader,
            'pocket_indices_loader': pocket_indices_loader,
            'protein_repr_loader': protein_repr_loader,
            'ligand_repr_loader': ligand_repr_loader,
            'morgan_repr_loader': morgan_repr_loader,
            'unimol_repr_loader': unimol_repr_loader,
            'label_loader': label_loader,
            'id_list': id_list,
        }

        # Add optional parameters if present
        optional_params = [
            'prob', 
            'overfit',
            'type'
        ]
        
        for param in optional_params:
            if hasattr(dataset_config, param):
                dataset_kwargs[param] = getattr(dataset_config, param)

        return DatasetResources(**dataset_kwargs)

    def _load_datasets_from_configs(self, dataset_configs, dataset_type: str = "train") -> list[DatasetResources]:
        """Load multiple datasets from configurations.
        
        Parameters
        ----------
        dataset_configs : list
            List of dataset configuration objects
        dataset_type : str
            Type of datasets ("train" or "validation")
            
        Returns
        -------
        list[DatasetResources]
            List of created datasets
        """
        datasets = []
        for dataset_config in dataset_configs:
            dataset = self._create_dataset_from_config(dataset_config, dataset_type)
            datasets.append(dataset)
        return datasets

    def worker_init_fn(self, worker_id):
        seed = torch.initial_seed() % (2**32) + worker_id
        np.random.seed(seed)

    def train_dataloader(self) -> DataLoader:
        dataset = AffinityTrainingDatasets(
            datasets=self.train_datasets,
            global_cfg=self.cfg,
        )

        print(f"Train dataset size: {len(dataset)}")

        # Balanced grouped sampler (online generation)
        samples_per_epoch = int(self.cfg.train_sets.get('samples_per_epoch', 0))
        if samples_per_epoch <= 0:
            raise ValueError("train_sets.samples_per_epoch must be > 0 for BalancedGroupBatchSampler")
        batch_size = int(self.cfg.train_sets.get('batch_size', 20))
        group_size = int(self.cfg.train_sets.get('group_size', 5))
        active_ratio = float(self.cfg.train_sets.get('active_ratio', 0.2))

        batch_sampler = BalancedGroupBatchSampler(
            dataset=dataset,
            samples_per_epoch=samples_per_epoch,
            batch_size=batch_size,
            group_size=group_size,
            active_ratio=active_ratio,
        )

        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=self.cfg.train_sets.get('num_workers', 16),
            pin_memory=self.cfg.train_sets.get('pin_memory', True),
            collate_fn=GroupedCollate(group_size),
            worker_init_fn=self.worker_init_fn,
        )

    def val_dataloader(self) -> DataLoader:
        if not self.val_datasets:
            print("Warning: No validation datasets found")
            return DataLoader(
                [],
                batch_size=1,
                num_workers=0,
                pin_memory=True,
                shuffle=False,
            )
            
        dataset = AffinityValidationDatasets(
            datasets=self.val_datasets,
            global_cfg=self.cfg,
        )

        print(f"Validation dataset size: {len(dataset)}")

        sampler = None
        if dist.is_available() and dist.is_initialized():
            sampler = DistributedSampler(dataset, shuffle=False)

        return DataLoader(
            dataset,
            batch_size=self.cfg.val_sets.get('batch_size', 5),
            sampler=sampler,
            num_workers=self.cfg.val_sets.get('num_workers', 16),
            pin_memory=self.cfg.val_sets.get('pin_memory', True),
            shuffle=False,
            collate_fn=collate_affinity,
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
                      'edge_index', 'edge_types', 'label', 'label_type', 'record_id']:
                batch[key] = batch[key].to(device)
        return batch