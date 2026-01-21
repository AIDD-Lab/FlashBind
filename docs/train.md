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

#### Resource Key Conventions

The following table shows the expected keys for each resource type:

| Resource | SDF Mode Key | Complex Mode Key |
|----------|--------------|------------------|
| `structure` | `prot_id` | `{prot_id}_{ligand_id}` |
| `ligand` | `{prot_id}_{ligand_id}` | `ligand_id` |
| `protein_repr` | `prot_id` | `prot_id` |
| `ligand_repr` | `ligand_id` | `ligand_id` |
| `pocket_indices` | `{prot_id}_{ligand_id}` | `{prot_id}_{ligand_id}` |

Where `{prot_id}_{ligand_id}` corresponds to the sample ID format in `id_list`.

**Note:** In SDF mode, `structure` contains only the protein and is shared across ligands, so it uses `prot_id` as the key. In complex mode, `structure` contains the protein-ligand complex, so it uses the full sample ID. The `ligand` field in SDF mode stores docking poses (one per protein-ligand pair), while in complex mode it stores SMILES strings (shared across proteins).

#### Atom Order Alignment in SDF Mode

When using `ligand_type=sdf`, the atom order in SDF files must match the order in `ligand_repr`. By default, this pipeline assumes SDF files are generated from FABind+ docking, which preserves the canonical SMILES atom order.

If your SDF files come from other sources, you must manually ensure the atom order matches `ligand_repr`. See `src/affinity/data/repr/torchdrug.py` for the `read_smiles` function that defines our canonical ordering.

**Important:** Applying similar canonicalization directly to SDF molecules (via `MolToSmiles` + `RenumberAtoms`) may produce different results than canonicalizing from SMILES. This is because RDKit's canonical ranking can be influenced by additional information in SDF files (e.g., 3D coordinates, bond storage order), leading to different tie-breaking decisions for symmetric atoms.

If you cannot guarantee atom order alignment, use the complex mode instead (`ligand_type=smiles` with protein-ligand complex structures). Complex mode uses substructure matching to explicitly align coordinates to the canonical SMILES order.

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
