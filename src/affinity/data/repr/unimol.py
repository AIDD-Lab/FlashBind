#############################################
# Use UniMol to generate ligand representations.
# Returns molecular-level embeddings [H] instead of atom-level [L, H].
#############################################

import warnings
import torch
import json
import lmdb
import pickle
import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from rdkit import Chem

try:
    from unimol_tools import UniMolRepr
except ImportError:
    raise ImportError("unimol_tools library not found. Please install it with: pip install unimol_tools")

# Global variables to store model
_model = None
_device = None
def initialize_unimol(model_name="unimolv2", model_size="1.1B", remove_hs=False, device="auto"):
    """Initialize UniMol model."""
    global _model, _device
    
    if device == "auto":
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        _device = torch.device(device)
    
    print(f"Loading UniMol model: {model_name} ({model_size})")
    print(f"Using device: {_device}")
    
    _model = UniMolRepr(
        data_type='molecule',
        remove_hs=remove_hs,
        model_name=model_name,
        model_size=model_size,
        use_ddp=True
    )
    
    return _model, _device

def extract_unimol_features_batch(smiles_list):
    """Extract UniMol features from a batch of SMILES strings."""
    global _model, _device
    
    if _model is None:
        raise RuntimeError("UniMol model not initialized. Call initialize_unimol() first.")
    
    try:
        # Get UniMol representations for the batch
        # return_atomic_reprs=False means we get CLS token representation (molecular-level)
        unimol_repr = _model.get_repr(smiles_list, return_atomic_reprs=False)
        # Convert to numpy array and then to torch tensor
        features = np.array(unimol_repr)  # [batch_size, hidden_dim]
        features_tensor = torch.from_numpy(features).float()
        assert features_tensor.shape[0] == len(smiles_list), f"Features shape: {features_tensor.shape[0]} != {len(smiles_list)}"
        return features_tensor.cpu()  # Ensure features are on CPU for consistent behavior
        
    except Exception as e:
        print(f"Batch processing failed: {str(e)}, falling back to single processing...")
        
        # Fallback to single processing
        results = []
        hidden_dim = None
        
        for smiles in smiles_list:
            try:
                unimol_repr = _model.get_repr([smiles], return_atomic_reprs=False)
                features = np.array(unimol_repr)
                features_tensor = torch.from_numpy(features).float().squeeze(0)  # [H]
                results.append(features_tensor)
                if hidden_dim is None:
                    hidden_dim = features_tensor.shape[0]
            except Exception as e2:
                print(f"Failed to process single SMILES: {smiles}, error: {str(e2)}")
                results.append(None)  # Mark as failed
        
        # If all failed, use default hidden_dim
        if hidden_dim is None:
            hidden_dim = 768  # Default hidden dimension
            print(f"Warning: All SMILES in batch failed, using default hidden_dim={hidden_dim}")
        
        # Replace failed ones with zero tensors
        final_results = []
        for r in results:
            if r is None:
                final_results.append(torch.zeros(hidden_dim))
            else:
                final_results.append(r)
        
        return torch.stack(final_results).cpu()

def process_smiles_batch(smiles_batch):
    """Process a batch of SMILES strings and return feature vectors with [H] shape."""
    # Get UniMol embeddings for the entire batch
    batch_embeddings = extract_unimol_features_batch(smiles_batch)
    if batch_embeddings is None:
        # print(f"Warning: Failed to process SMILES batch with {len(smiles_batch)} SMILES: {smiles_batch}")
        return [(smiles, None) for smiles in smiles_batch]
    
    # Process each SMILES and return [H] shape embedding
    results = []
    for i, smiles in enumerate(smiles_batch):
        smiles = smiles.strip()
        # Directly use the embedding without expansion
        embedding = batch_embeddings[i].clone()  # [H]
        results.append((smiles, embedding))
    
    assert len(results) == len(smiles_batch), f"Results length: {len(results)} != {len(smiles_batch)}"
    return results

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate ligand representations using UniMol from a SMILES JSON file.")
    parser.add_argument("--input_json", default="./data/ESIBank/DUF/smiles.json", 
                       help="Path to the input JSON file (ligand_id -> SMILES)")
    parser.add_argument("--output_lmdb", default="./data/ESIBank/DUF/repr/unimol.lmdb", 
                       help="Path to save the output LMDB database")
    parser.add_argument("--model_name", default="unimolv2", choices=["unimolv1", "unimolv2"],
                       help="UniMol model name")
    parser.add_argument("--model_size", default="84m", choices=["84m", "164m", "310m", "570m", "1.1B"],
                       help="Model size (only for unimolv2)")
    parser.add_argument("--remove_hs", action="store_true", default=False,
                       help="Whether to remove hydrogen atoms")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                       help="Device to use for computation")
    parser.add_argument("--batch_size", type=int, default=1024, 
                       help="Batch size for processing SMILES")
    args = parser.parse_args()
    
    print(f"Reading JSON file: {args.input_json}")
    try:
        with open(args.input_json, 'r') as f:
            ligand_data = json.load(f)
        print(f"Loaded {len(ligand_data)} ligand entries.")
    except Exception as e:
        print(f"Error: Failed to read or parse JSON file: {e}")
        return

    # --- Resume: Check existing LMDB ---
    existing_ids = set()
    if os.path.exists(args.output_lmdb):
        print(f"Found existing LMDB: {args.output_lmdb}, loading existing keys...")
        env = lmdb.open(args.output_lmdb, readonly=True, lock=False)
        with env.begin() as txn:
            cursor = txn.cursor()
            for key, _ in cursor:
                existing_ids.add(key.decode('utf-8'))
        env.close()
        print(f"Found {len(existing_ids)} existing ligand entries.")
    
    # Filter out already processed ligands
    ligand_data_filtered = {k: v for k, v in ligand_data.items() if k not in existing_ids}
    print(f"Remaining ligands to process: {len(ligand_data_filtered)} (skipped {len(existing_ids)} existing)")
    
    if len(ligand_data_filtered) == 0:
        print("All ligands already processed. Exiting.")
        return

    # Initialize UniMol model
    initialize_unimol(
        model_name=args.model_name, 
        model_size=args.model_size,
        remove_hs=args.remove_hs,
        device=args.device
    )

    # --- De-duplication Step ---
    print("Grouping ligands by unique SMILES string...")
    smiles_to_ids = defaultdict(list)
    for ligand_id, smiles in ligand_data_filtered.items():
        smiles_to_ids[smiles.strip()].append(ligand_id)

    unique_smiles = list(smiles_to_ids.keys())
    print(f"Found {len(unique_smiles)} unique SMILES strings to process.")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_lmdb), exist_ok=True)
    
    # Process unique SMILES in batches
    print(f"Processing {len(unique_smiles)} unique SMILES sequences in batches of {args.batch_size}...")
    results = []

    for i in tqdm(range(0, len(unique_smiles), args.batch_size), desc="Processing SMILES batches"):
        batch = unique_smiles[i:i + args.batch_size]
        batch_results = process_smiles_batch(batch)
        results.extend(batch_results)
    
    print(f"Processed {len(results)} SMILES sequences.")
    # Map successful results back to a dictionary for easy lookup
    # --- Re-assign features to original ligand_ids ---
    final_results = []
    for smiles, features in results:
        if features is None:
            print(f"Warning: Skipping invalid SMILES entry for smiles '{smiles}': {features}")
            continue
        for ligand_id in smiles_to_ids[smiles]:
            final_results.append((ligand_id, features))
    
    processed_count = len(final_results)
    failed_count = len(ligand_data) - processed_count
    print(f"\nSuccessfully processed: {processed_count} ligands")
    print(f"Failed to process: {failed_count} ligands")
    
    if processed_count == 0:
        print("No ligands were successfully processed. Exiting.")
        return
    
    # Create/Open LMDB database (append mode for resume)
    print(f"Writing to LMDB database: {args.output_lmdb}")
    # Estimate map size based on feature dimensions (UniMol molecular-level embeddings)
    total_count = len(existing_ids) + processed_count
    map_size = max(total_count * 10 * 1024 * 1024, 1024 * 1024 * 1024)  # At least 1GB
    env = lmdb.open(args.output_lmdb, map_size=map_size)
    
    # Write to LMDB
    with env.begin(write=True) as txn:
        for ligand_id, features in tqdm(final_results, desc="Writing to LMDB"):
            features_bytes = pickle.dumps(features)
            key = str(ligand_id).encode('utf-8')
            txn.put(key, features_bytes)
            
    env.close()
    print(f"LMDB database updated successfully at: {args.output_lmdb}")
    print(f"New ligands added: {len(final_results)}")
    print(f"Total ligands stored: {len(existing_ids) + len(final_results)}")
    
    # Print feature shape info
    if final_results:
        sample_features = final_results[0][1]
        print(f"Feature shape per ligand: {sample_features.shape}")
        print(f"Feature dimension: {sample_features.numel()}")

if __name__ == "__main__":
    main()
