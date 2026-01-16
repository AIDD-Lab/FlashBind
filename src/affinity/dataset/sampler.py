"""
Affinity Training Sampler

This module contains the sampler for affinity training that ensures balanced
sampling of active and inactive compounds in grouped batches.
"""

from typing import Iterator, Dict, List, Tuple
from collections import defaultdict

import os
import time
import numpy as np
from torch.utils.data import Sampler
from affinity.utils import extract_ids
import torch.distributed as dist


class BalancedGroupBatchSampler(Sampler[List[int]]):
    """
    Batch sampler that builds each batch online as groups of size `group_size` from the same protein,
    with an active:inactive ratio defined by `active_ratio`. Supports multiple datasets with per-dataset
    probabilities, and DDP distribution without overlap across ranks. Designed to be used as
    DataLoader.batch_sampler (not sampler+batch_size).
    """

    def __init__(
        self,
        dataset,  # AffinityTrainingDatasets (combined dataset)
        samples_per_epoch: int,
        batch_size: int,
        group_size: int = 5,
        active_ratio: float = 0.2,
    ) -> None:
        if batch_size % group_size != 0:
            raise ValueError(f"batch_size {batch_size} must be a multiple of group_size {group_size}")

        # Rank/world size
        if dist.is_available() and dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

        self.dataset = dataset
        self.samples_per_epoch = int(samples_per_epoch)
        self.batch_size = int(batch_size)
        self.group_size = int(group_size)
        # Ensure at least 1 active and at least 0 inactive
        self.active_per_group = max(1, int(round(self.group_size * float(active_ratio))))
        self.active_per_group = min(self.active_per_group, self.group_size - 1) if self.group_size > 1 else 1
        self.inactive_per_group = self.group_size - self.active_per_group

        # Reorganized storage: group by dataset type, eliminate placeholders
        self.binary_datasets: Dict[int, Dict] = {}  # dataset_idx -> {active_pools, inactive_pools, valid_proteins}
        self.value_datasets: Dict[int, Dict] = {}   # dataset_idx -> {value_pools, protein_weights, valid_proteins}
        self.enzyme_datasets: Dict[int, Dict[str, List[str]]] = {}  

        # Dataset metadata
        self.dataset_info: Dict[int, Dict] = {}  # dataset_idx -> {type, prob, size}

        all_ids = np.array(self.dataset.all_ids, dtype=object)
        dataset_indices = np.array(self.dataset.dataset_indices, dtype=np.int64)

        valid_dataset_indices = []
        probs = []
        
        for dataset_idx, ds in enumerate(self.dataset.datasets):
            mask = dataset_indices == dataset_idx
            included_ids = all_ids[mask]
            dataset_type = ds.type # Default to binary for backward compatibility
            size = 0
            
            if dataset_type == "binary":
                active: Dict[str, List[str]] = defaultdict(list)
                inactive: Dict[str, List[str]] = defaultdict(list)

                for sample_id in included_ids:
                    try:
                        label_data = ds.label_loader.get(sample_id)
                        label_value = label_data["label"][0] if isinstance(label_data["label"], list) else label_data["label"]
                        is_active = label_value.lower() == "true"
                        prot_id, _ = extract_ids(sample_id)
                        if is_active:
                            active[prot_id].append(sample_id)
                        else:
                            inactive[prot_id].append(sample_id)
                    except Exception:
                        import traceback
                        traceback.print_exc()
                        continue

                valid_proteins = [
                    pid for pid in active.keys()
                    if len(active[pid]) > 0 and len(inactive.get(pid, [])) > 0
                ]
                
                # Skip datasets with no valid proteins
                if len(valid_proteins) == 0:
                    continue
                    
                size = sum(len(samples) for samples in active.values()) + sum(len(samples) for samples in inactive.values())
                
                # Store binary dataset info
                self.binary_datasets[dataset_idx] = {
                    'active_pools': active,
                    'inactive_pools': inactive,
                    'valid_proteins': np.array(valid_proteins, dtype=object)
                }
                
            elif dataset_type == "value":
                # Collect samples and labels for IQR calculation
                protein_samples: Dict[str, List[str]] = defaultdict(list)
                protein_labels: Dict[str, List[float]] = defaultdict(list)
                
                for sample_id in included_ids:
                    try:
                        label_data = ds.label_loader.get(sample_id)
                        label_value = label_data["label"][0] if isinstance(label_data["label"], list) else label_data["label"]
                        prot_id, _ = extract_ids(sample_id)
                        protein_samples[prot_id].append(sample_id)
                        protein_labels[prot_id].append(label_value)
                    except Exception:
                        import traceback
                        traceback.print_exc()
                        continue
                
                # Calculate IQR weights for each protein
                protein_weights: Dict[str, float] = {}
                value_pools: Dict[str, List[str]] = {}
                
                for prot_id in protein_samples.keys():
                    if len(protein_samples[prot_id]) >= 2:  # At least need 2 samples for quartiles
                        labels = protein_labels[prot_id]
                        q1, q3 = np.percentile(labels, [25, 75])
                        iqr_weight = abs(q3 - q1)
                        protein_weights[prot_id] = iqr_weight
                        value_pools[prot_id] = protein_samples[prot_id]
                
                valid_proteins = list(protein_weights.keys())
                
                # Skip datasets with no valid proteins
                if len(valid_proteins) == 0:
                    continue
                    
                size = sum(len(samples) for samples in value_pools.values())
                
                # Store value dataset info
                self.value_datasets[dataset_idx] = {
                    'value_pools': value_pools,
                    'protein_weights': protein_weights,
                    'valid_proteins': np.array(valid_proteins, dtype=object)
                }
                
            elif dataset_type == "enzyme":
                # For enzyme datasets, collect samples and classify as active/inactive
                enzyme_active = []
                enzyme_inactive = []
                
                for sample_id in included_ids:
                    try:
                        label_data = ds.label_loader.get(sample_id)
                        label_value = label_data["label"][0] if isinstance(label_data["label"], list) else label_data["label"]
                        is_active = label_value.lower() == "true"
                        
                        if is_active:
                            enzyme_active.append(sample_id)
                        else:
                            enzyme_inactive.append(sample_id)
                    except Exception:
                        import traceback
                        traceback.print_exc()
                        continue
                
                # Skip datasets with no samples
                if len(enzyme_active) == 0 and len(enzyme_inactive) == 0:
                    continue
                
                size = len(enzyme_active) + len(enzyme_inactive)

                # Add to global enzyme pools for efficient cross-dataset sampling
                self.enzyme_datasets[dataset_idx] = {
                    'active': enzyme_active,
                    'inactive': enzyme_inactive
                }


            else:
                raise ValueError(f"Unsupported dataset type: {dataset_type}")
                
            # Store dataset metadata and calculate dataset size
            prob = getattr(ds, 'prob', None)
            if prob is not None:
                prob = max(0.0, float(prob))
            
            self.dataset_info[dataset_idx] = {
                'type': dataset_type,
                'prob': prob,
                'size': size,
            }
            
            valid_dataset_indices.append(dataset_idx)
            probs.append(prob)

        if not valid_dataset_indices:
            raise ValueError("No datasets have valid proteins")

        # Calculate probabilities with None handling
        explicit_probs = [p for p in probs if p is not None]
        none_indices = [i for i, p in enumerate(probs) if p is None]
        
        if not explicit_probs:
            # All None, use size-based allocation
            total_size = sum(self.dataset_info[idx]['size'] for idx in valid_dataset_indices)
            final_probs = [self.dataset_info[idx]['size'] / total_size for idx in valid_dataset_indices]
        else:
            explicit_sum = sum(explicit_probs)
            if explicit_sum > 1.0 or explicit_sum < 0.0:
                print(f"Warning: Explicit probabilities sum to {explicit_sum}, falling back to size-based allocation")
                total_size = sum(self.dataset_info[idx]['size'] for idx in valid_dataset_indices)
                final_probs = [self.dataset_info[idx]['size'] / total_size for idx in valid_dataset_indices]
            else:
                # Distribute remaining probability by size
                remaining_prob = 1.0 - explicit_sum
                none_total_size = sum(self.dataset_info[valid_dataset_indices[i]]['size'] for i in none_indices)
                
                final_probs = []
                for i, prob in enumerate(probs):
                    if prob is not None:
                        final_probs.append(prob)
                    else:
                        size_ratio = self.dataset_info[valid_dataset_indices[i]]['size'] / none_total_size
                        final_probs.append(remaining_prob * size_ratio)
        
        self.dataset_probs = np.array(final_probs, dtype=np.float64)
        print(f"Dataset probabilities: {self.dataset_probs}")
        self.valid_dataset_indices = valid_dataset_indices

        # Build local_to_global mapping with new dataset indices
        self.local_to_global: Dict[Tuple[str, int], int] = {}
        for global_id, (sample_id, dataset_index) in enumerate(zip(self.dataset.all_ids, self.dataset.dataset_indices)):
            self.local_to_global[(sample_id, dataset_index)] = global_id

        # Initialize RNG for sampling
        self.rng = np.random.default_rng()

    def __len__(self) -> int:
        global_batches = self.samples_per_epoch // self.batch_size
        return max(0, global_batches // self.world_size)

    def __iter__(self) -> Iterator[List[int]]:
        total_batches = len(self)
        num_groups = self.batch_size // self.group_size

        for _ in range(total_batches):
            # Choose dataset by probability
            chosen_idx = int(self.rng.choice(len(self.valid_dataset_indices), p=self.dataset_probs))
            dataset_idx = self.valid_dataset_indices[chosen_idx]
            dataset_type = self.dataset_info[dataset_idx]['type']

            batch_indices: List[int] = []
            
            if dataset_type == "enzyme":
                active_count = num_groups * self.active_per_group
                inactive_count = num_groups * self.inactive_per_group
                
                # Sample active samples
                for _ in range(active_count):
                    while True:
                        idx = int(self.rng.choice(len(self.valid_dataset_indices), p=self.dataset_probs))
                        ds_idx = self.valid_dataset_indices[idx]
                        if ds_idx in self.enzyme_datasets and len(self.enzyme_datasets[ds_idx]['active']) > 0:
                            break
                    sample_id = self.rng.choice(self.enzyme_datasets[ds_idx]['active'])
                    batch_indices.append(self.local_to_global[(sample_id, ds_idx)])
                
                # Sample inactive samples
                for _ in range(inactive_count):
                    while True:
                        idx = int(self.rng.choice(len(self.valid_dataset_indices), p=self.dataset_probs))
                        ds_idx = self.valid_dataset_indices[idx]
                        if ds_idx in self.enzyme_datasets and len(self.enzyme_datasets[ds_idx]['inactive']) > 0:
                            break
                    sample_id = self.rng.choice(self.enzyme_datasets[ds_idx]['inactive'])
                    batch_indices.append(self.local_to_global[(sample_id, ds_idx)])
                    
            elif dataset_type == "binary":
                # Binary dataset: use stored binary dataset info
                binary_info = self.binary_datasets[dataset_idx]
                valid_proteins = binary_info['valid_proteins']
                active_pools = binary_info['active_pools']
                inactive_pools = binary_info['inactive_pools']
                
                # Sample proteins for each group (with replacement)
                proteins = self.rng.choice(valid_proteins, size=num_groups, replace=True)
                
                # Process protein-based sampling for binary datasets
                for prot_id in proteins:
                    # Active
                    active_pool = active_pools[prot_id]
                    if len(active_pool) >= self.active_per_group:
                        active_ids = list(self.rng.choice(active_pool, size=self.active_per_group, replace=False))
                    else:
                        active_ids = list(self.rng.choice(active_pool, size=self.active_per_group, replace=True))

                    # Inactive
                    inactive_pool = inactive_pools[prot_id]
                    if len(inactive_pool) >= self.inactive_per_group:
                        inactive_ids = list(self.rng.choice(inactive_pool, size=self.inactive_per_group, replace=False))
                    else:
                        inactive_ids = list(self.rng.choice(inactive_pool, size=self.inactive_per_group, replace=True))

                    assert len(active_ids) + len(inactive_ids) == self.group_size, f"Group size mismatch: {len(active_ids)} + {len(inactive_ids)} != {self.group_size}"
                    
                    # Preserve group order: active first then inactive
                    group_ids = active_ids + inactive_ids

                    # Convert sample_ids to global indices
                    for sample_id in group_ids:
                        global_idx = self.local_to_global[(sample_id, dataset_idx)]
                        batch_indices.append(global_idx)
                        
            elif dataset_type == "value":
                # Value dataset: use stored value dataset info
                value_info = self.value_datasets[dataset_idx]
                valid_proteins = value_info['valid_proteins']
                value_pools = value_info['value_pools']
                protein_weights = value_info['protein_weights']
                
                # IQR-weighted protein sampling
                weights = np.array([protein_weights.get(pid, 0.0) for pid in valid_proteins])
                if weights.sum() > 0:
                    weights = weights / weights.sum()  # Normalize weights
                    proteins = self.rng.choice(valid_proteins, size=num_groups, replace=True, p=weights)
                else:
                    proteins = self.rng.choice(valid_proteins, size=num_groups, replace=True)
                
                # Process protein-based sampling for value datasets
                for prot_id in proteins:
                    value_pool = value_pools[prot_id]  # List of sample_ids
                    
                    # Uniform sampling: no replacement if enough samples, otherwise with replacement
                    if len(value_pool) >= self.group_size:
                        group_ids = list(self.rng.choice(value_pool, size=self.group_size, replace=False))
                    else:
                        group_ids = list(self.rng.choice(value_pool, size=self.group_size, replace=True))

                    assert len(group_ids) == self.group_size, f"Group size mismatch: {len(group_ids)} != {self.group_size}"

                    # Convert sample_ids to global indices
                    for sample_id in group_ids:
                        global_idx = self.local_to_global[(sample_id, dataset_idx)]
                        batch_indices.append(global_idx)
                        
            else:
                raise ValueError(f"Unsupported dataset type: {dataset_type}")

            assert len(batch_indices) == self.batch_size, f"Batch size mismatch: {len(batch_indices)} != {self.batch_size}"
            yield batch_indices