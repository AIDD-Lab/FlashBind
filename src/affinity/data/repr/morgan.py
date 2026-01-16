#######################################################
# Generate Morgan fingerprints from SMILES and store in LMDB.
# Modified based on user's torchdrug script and morgan fp code.
#######################################################

import lmdb
import pickle
import os
import json
import numpy as np
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, rdFingerprintGenerator 
from multiprocessing import cpu_count
from joblib import Parallel, delayed
from tqdm import tqdm


def get_morgan_feature(mol):
    """
    From RDKit Mol to generate Morgan fingerprint.
    """
    GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fp = GENERATOR.GetFingerprint(mol)
    arr = np.zeros((0,), dtype=np.int8) 
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

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

# --- Modified processing function for unique SMILES ---
def process_unique_smiles(smiles):
    try:
        mol = read_smiles(smiles)
        if mol is None:
            print(f"Warning: Failed to parse SMILES: {smiles}")
            return smiles, None
        
        features = get_morgan_feature(mol)
        return smiles, features
    except Exception as e:
        print(f"Error processing SMILES {smiles}: {str(e)}")
        return smiles, None

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate ligand Morgan fingerprints from a SMILES JSON file.")
    parser.add_argument("--input_json", default="./data/mf-pcba/smiles.json", help="Path to the input JSON file (ligand_id -> SMILES)")
    parser.add_argument("--output_lmdb", default="./data/mf-pcba/repr/morgan.lmdb", help="Path to save the output LMDB database")
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
    
    if args.n_jobs == -1:
        n_jobs = cpu_count()
    else:
        n_jobs = args.n_jobs
    print(f"Using {n_jobs} parallel jobs.")

    # Process unique SMILES in parallel
    print("Processing unique SMILES sequences in parallel...")
    results = Parallel(n_jobs=n_jobs)(
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
    total_ligands = sum(len(ids) for ids in smiles_to_ids.values())
    failed_count = total_ligands - processed_count
    
    print(f"\nSuccessfully processed: {processed_count} ligands")
    print(f"Failed to process: {failed_count} ligands")
    
    # Create LMDB database
    print(f"Creating LMDB database: {args.output_lmdb}")
    map_size_estimate = max(processed_count * 10 * 1024 * 1024, 1024 * 1024 * 1024) 
    print(f"LMDB map_size estimated: {map_size_estimate / (1024*1024*1024):.2f} GB")
    env = lmdb.open(args.output_lmdb, map_size=map_size_estimate)
    
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