# Training Guide

This document describes how to train the model and explains the key configuration parameters.

## Quick Start

### Training

```bash
python ./scripts/train.py ./scripts/configs/affinity_binary.yaml
```

### Debug

```bash
python ./scripts/train.py ./scripts/configs/affinity_binary.yaml debug=1
```

In debug mode:
- Wandb logging is disabled
- Uses single GPU only
- Sets `num_workers=0`

## Configuration Parameters

All parameters can be modified in the YAML configuration file or overridden via command line.

### Dataset Configuration

Each dataset entry in `data.train_sets.datasets` or `data.val_sets.datasets` supports the following fields:

| Parameter | Description |
|-----------|-------------|
| `type` | Dataset type: `binary`, `value`, or `enzyme`. Each type has different sampler settings. |
| `structure` | Path to structure files. |
| `structure_type` | Structure format: `pdb` or `cif`. |
| `ligand` | Path to ligand files. |
| `ligand_type` | Ligand format: `sdf` or `smiles`. |
| `pocket_indices` | (Optional) List of pocket residue indices. |
| `protein_repr` | Protein representation file (e.g., ESM3 features). |
| `ligand_repr` | Ligand representation file (e.g., TorchDrug features). |
| `label` | Label file. |
| `id_list` | Sample ID list. |

#### Structure Input Modes

The `structure` field behaves differently depending on `ligand_type`:

- **When `ligand_type=sdf`**: The `structure` contains only the protein structure.
- **When `ligand_type=smiles`**: The `structure` contains the protein-ligand complex structure. In this case, ligand bond information is determined from the SMILES string.

#### Pocket Indices

The `pocket_indices` field is optional:
- If not provided, or if the pre-computed pocket is too large, the cropper will automatically crop the pocket based on spatial distance.

#### Supported Resource Formats

Input files support multiple formats including `lmdb`, `pt`, `json`, etc. See `src/affinity/utils/resource_loader.py` for details.

### Training Arguments

The `model.training_args` section controls optimizer and scheduler settings.

#### Optimizers

| Optimizer | Description |
|-----------|-------------|
| `adamw` | Standard AdamW optimizer for all parameters (default). |
| `muon` | Muon optimizer for encoder hidden weights (ndim ≥ 2), AdamW for the rest. |

#### Learning Rate Schedulers

| Scheduler | Description |
|-----------|-------------|
| `cosine_restart` | Cosine annealing with warm restarts (default). |
| `cosine` | Standard cosine annealing. |
| `linear_decay` | Linear learning rate decay. |
| `constant` | Constant learning rate. |
| `reduce_on_plateau` | Reduce LR when a metric stops improving. |

For detailed scheduler parameters, see `src/affinity/model/model.py`.
