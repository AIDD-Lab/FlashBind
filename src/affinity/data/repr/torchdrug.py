#############################################
# Use torchdrug to generate ligand representations.
# Code from source code of torchdrug.
#############################################

import warnings
import torch
import pandas as pd
import lmdb
import pickle
import os
import json
from collections import defaultdict
from rdkit import Chem
from multiprocessing import cpu_count
from joblib import Parallel, delayed
from tqdm import tqdm

# --- Vocabularies ---
atom_vocab = ["H", "B", "C", "N", "O", "F", "Mg", "Si", "P", "S", "Cl", "Cu", "Zn", "Se", "Br", "Sn", "I"]
atom_vocab = {a: i for i, a in enumerate(atom_vocab)}
degree_vocab = range(7)
num_hs_vocab = range(7)
formal_charge_vocab = range(-5, 6)
total_valence_vocab = range(8)
bond_type_vocab = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                   Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
bond_type_vocab = {b: i for i, b in enumerate(bond_type_vocab)}
bond_dir_vocab = range(len(Chem.rdchem.BondDir.values))
bond_stereo_vocab = range(len(Chem.rdchem.BondStereo.values))
residue_vocab = ["GLY", "ALA", "SER", "PRO", "VAL", "THR", "CYS", "ILE", "LEU", "ASN",
                 "ASP", "GLN", "LYS", "GLU", "MET", "HIS", "PHE", "ARG", "TYR", "TRP"]

# --- Featurization Functions ---
def onehot(x, vocab, allow_unknown=False):
    if x in vocab:
        if isinstance(vocab, dict):
            index = vocab[x]
        else:
            index = vocab.index(x)
    else:
        index = -1
    if allow_unknown:
        feature = [0] * (len(vocab) + 1)
        if index == -1:
            warnings.warn("Unknown value `%s`" % x)
        feature[index] = 1
    else:
        feature = [0] * len(vocab)
        if index == -1:
            raise ValueError("Unknown value `%s`. Available vocabulary is `%s`" % (x, vocab))
        feature[index] = 1
    return feature

def atom_property_prediction(atom):
    return onehot(atom.GetSymbol(), atom_vocab, allow_unknown=True) + \
           onehot(atom.GetDegree(), degree_vocab, allow_unknown=True) + \
           onehot(atom.GetTotalNumHs(), num_hs_vocab, allow_unknown=True) + \
           onehot(atom.GetTotalValence(), total_valence_vocab, allow_unknown=True) + \
           onehot(atom.GetFormalCharge(), formal_charge_vocab, allow_unknown=True) + \
           [atom.GetIsAromatic()]

def read_smiles(smile):
    try:
        mol = Chem.MolFromSmiles(smile)
    except:
        print("warning: cannot sanitize smiles: ", smile)
        mol = Chem.MolFromSmiles(smile, sanitize=False)
    
    _ = Chem.MolToSmiles(mol)
    m_order = list(mol.GetPropsAsDict(includePrivate=True, includeComputed=True)['_smilesAtomOutputOrder'])
    mol = Chem.RenumberAtoms(mol, m_order)
    mol = Chem.RemoveAllHs(mol, sanitize=False)
    Chem.SanitizeMol(mol)
    return mol

def extract_feature(mol):
    all_atom_features = []
    for atom in mol.GetAtoms():
        all_atom_features.append(atom_property_prediction(atom))
    return torch.tensor(all_atom_features, dtype=torch.float)

# --- Modified processing function for unique SMILES ---
def process_unique_smiles(smiles):
    """Process a single unique SMILES string and return the feature vector."""
    try:
        mol = read_smiles(smiles)
        if mol is None:
            print(f"Warning: Failed to parse SMILES: {smiles}")
            return smiles, None
        
        features = extract_feature(mol)
        return smiles, features
    except Exception as e:
        # This will be printed in the worker process
        print(f"Error processing SMILES {smiles}: {str(e)}")
        return smiles, None

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate ligand representations from a SMILES JSON file.")
    parser.add_argument("--input_json", default="./data/mf-pcba/smiles.json", help="Path to the input JSON file (ligand_id -> SMILES)")
    parser.add_argument("--output_lmdb", default="./data/mf-pcba/repr/torchdrug.lmdb", help="Path to save the output LMDB database")
    parser.add_argument("--n_jobs", type=int, default=-1, help="Number of parallel jobs (-1 for all available cores)")
    args = parser.parse_args()
    
    print(f"Reading JSON file: {args.input_json}")
    try:
        with open(args.input_json, 'r') as f:
            ligand_data = json.load(f)
        print(f"Loaded {len(ligand_data)} ligand entries.")
    except Exception as e:
        print(f"Error: Failed to read or parse JSON file: {e}")
        return

    # --- De-duplication Step ---
    print("Grouping ligands by unique SMILES string...")
    smiles_to_ids = defaultdict(list)
    for ligand_id, smiles in ligand_data.items():
        if smiles and isinstance(smiles, str):
            smiles_to_ids[smiles].append(ligand_id)
        else:
            print(f"Warning: Skipping invalid SMILES entry for ligand_id '{ligand_id}': {smiles}")

    unique_smiles = list(smiles_to_ids.keys())
    print(f"Found {len(unique_smiles)} unique SMILES strings to process.")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_lmdb), exist_ok=True)
    print("cpu_count():", cpu_count())
    # Process unique SMILES in parallel
    print("Processing unique SMILES sequences in parallel...")
    results = Parallel(n_jobs=args.n_jobs)(
        delayed(process_unique_smiles)(smiles) 
        for smiles in tqdm(unique_smiles, desc="Processing unique SMILES")
    )
    
    # Map successful results back to a dictionary for easy lookup
    smiles_to_features = {smiles: features for smiles, features in results if features is not None}
    
    # --- Re-assign features to original ligand_ids ---
    final_results = []
    for smiles, ligand_ids in smiles_to_ids.items():
        if smiles in smiles_to_features:
            features = smiles_to_features[smiles]
            for ligand_id in ligand_ids:
                final_results.append((ligand_id, features))

    processed_count = len(final_results)
    failed_count = len(ligand_data) - processed_count
    print(f"\nSuccessfully processed: {processed_count} ligands")
    print(f"Failed to process: {failed_count} ligands")
    
    # Create LMDB database
    print(f"Creating LMDB database: {args.output_lmdb}")
    map_size = max(processed_count * 10 * 1024 * 1024, 1024 * 1024 * 1024)  # At least 1GB
    env = lmdb.open(args.output_lmdb, map_size=map_size)
    
    # Write to LMDB
    with env.begin(write=True) as txn:
        for ligand_id, features in tqdm(final_results, desc="Writing to LMDB"):
            features_bytes = pickle.dumps(features)
            key = str(ligand_id).encode('utf-8')
            txn.put(key, features_bytes)
            
    env.close()
    print(f"LMDB database created successfully at: {args.output_lmdb}")
    print(f"Total ligands stored: {len(final_results)}")

if __name__ == "__main__":
    main()