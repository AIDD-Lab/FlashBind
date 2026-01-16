# Data Processing Guide

This document describes the data format and preprocessing pipeline for BioAssay models.

## Input Data Format

Three JSON files are required as basic input:

| File | Description |
|------|-------------|
| `prots.json` | Mapping from `prot_id` to protein sequence |
| `smiles.json` | Mapping from `ligand_id` to SMILES string |
| `id.json` | List of sample IDs in format `{prot_id}_{ligand_id}` |

## Structure Prediction

### Protein Structure

Use `src/affinity/data/fold.py` to obtain protein structures:

1. The script first searches for existing structures in PDB and AlphaFoldDB.
2. If no structure is found, it uses **Boltz-2** to predict the structure.

### Ligand Structure (Docking)

We use **FABind+** to predict ligand binding poses. See `FABind_plus/README.md` for environment setup.

**Pipeline:**
1. Convert input data to FABind+ format
Before running, open FABind_plus/convert_data_to_csv.py and set the required parameters. Save the file and execute:

```bash
python FABind_plus/convert_data_to_csv.py
```

2. Run FABind+ inference (modify parameters in each script as needed):
   ```bash
   bash FABind_plus/run_molecule.sh
   bash FABind_plus/run_protein.sh
   bash FABind_plus/run.sh
   ```

3. The modified FABind+ outputs:
   - `ligand_sdf.lmdb`: Mapping from `{prot_id}_{ligand_id}` to ligand docking pose (SDF format)
   - `pocket_indices.lmdb`: Mapping from `{prot_id}_{ligand_id}` to pocket residue indices

## Feature Extraction

Feature extraction scripts are located in `src/affinity/data/repr/`:

| Script | Description |
|--------|-------------|
| `esm3.py` | Extract ESM3 protein representations |
| `torchdrug.py` | Extract TorchDrug ligand features |
| `morgan.py` | Generate Morgan fingerprints |
| `unimol.py` | Extract UniMol representations |

## Data Leakage Filtering

Scripts for handling data leakage are in `src/affinity/data/filter/`:

| Script | Description |
|--------|-------------|
| `mmseqs.py` | Filter proteins by sequence similarity |
| `tanimoto.py` | Filter ligands by Tanimoto similarity |
