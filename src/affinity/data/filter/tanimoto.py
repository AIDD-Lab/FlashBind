import json
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem import rdFingerprintGenerator
import warnings
warnings.filterwarnings('ignore')

########################################################
# This script filters ligands based on Tanimoto similarity to reference compounds
# 1. Load reference SMILES from multiple txt/json files and deduplicate
# 2. Load multiple datasets with id.json (ID lists) and smiles.json (ID->SMILES mapping)
# 3. Filter out ligands with Tanimoto similarity > 0.4 to any reference compound
# 4. Save filtered ID lists to final_id.json in each dataset directory
# 5. Output statistics for each dataset and overall summary
########################################################

# Global variable for reference SMILES files (to be filled by user)
REFERENCE_SMILES_FILES = [
    #"./data/mf-pcba/active_smiles.txt",
    "./data/casp16/smiles.json",
    "./data/openfe/smiles.json",
    "./data/fep4/smiles.json",
]

def load_reference_smiles(file_paths):
    """Load and deduplicate reference SMILES from multiple txt and json files"""
    if not file_paths:
        print("Warning: No reference SMILES files provided")
        return set()
    
    reference_smiles = set()
    total_loaded = 0
    
    print("Loading reference SMILES files...")
    for file_path in file_paths:
        path = Path(file_path)
        if not path.exists():
            print(f"Warning: Reference file not found: {file_path}")
            continue
            
        print(f"Reading {file_path}...")
        if path.suffix.lower() == '.json':
            # Load JSON file and extract SMILES from dict values
            with open(path, 'r') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    for smiles in data.values():
                        if smiles and isinstance(smiles, str):
                            reference_smiles.add(smiles)
                            total_loaded += 1
                else:
                    print(f"Warning: JSON file {file_path} should contain a dictionary")
        else:
            # Load txt file (original logic)
            with open(path, 'r') as f:
                for line in f:
                    smiles = line.strip()
                    if smiles:
                        reference_smiles.add(smiles)
                        total_loaded += 1
    
    print(f"Loaded {total_loaded} SMILES, {len(reference_smiles)} unique after deduplication")
    return reference_smiles

GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def compute_fingerprints(smiles_list):
    """Compute Morgan fingerprints for a list of SMILES"""
    fingerprints = []
    valid_smiles = []
    
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                fp = GENERATOR.GetFingerprint(mol)
                fingerprints.append(fp)
                valid_smiles.append(smiles)
        except:
            continue
    
    return fingerprints, valid_smiles

# Global variables for worker processes
_ref_fingerprints = None
_smiles_dict = None
_similarity_threshold = None

def init_worker(ref_fingerprints, smiles_dict, similarity_threshold):
    """Initialize worker process with shared data"""
    global _ref_fingerprints, _smiles_dict, _similarity_threshold
    _ref_fingerprints = ref_fingerprints
    _smiles_dict = smiles_dict
    _similarity_threshold = similarity_threshold


def filter_single_ligand_worker(ligand_id):
    """Worker function for filtering a single ligand"""
    global _ref_fingerprints, _smiles_dict, _similarity_threshold
    
    if ligand_id not in _smiles_dict:
        return (ligand_id, False)  # Remove if no SMILES
    
    smiles = _smiles_dict[ligand_id]
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return (ligand_id, False)  # Remove if invalid SMILES
        
        # Compute fingerprint
        ligand_fp = GENERATOR.GetFingerprint(mol)
        
        # Check similarity with reference compounds using bulk calculation
        similarities = DataStructs.BulkTanimotoSimilarity(ligand_fp, _ref_fingerprints)
        max_similarity = max(similarities) if similarities else 0
        # Keep if similarity <= threshold (i.e., not too similar to reference compounds)
        keep = max_similarity <= _similarity_threshold
    
        return (ligand_id, keep)
        
    except:
        return (ligand_id, False)  # Remove if any error

def filter_unique_ligands_parallel(sample_ids, ref_fingerprints, smiles_dict, similarity_threshold, n_processes):
    """Filter ligands in parallel and return keep/remove mapping for sample IDs"""
    
    # Step 1: Extract unique ligand IDs from sample IDs
    unique_ligand_ids = set()
    for sample_id in sample_ids:
        if '_' in sample_id:
            ligand_id = sample_id.split('_')[1]
            unique_ligand_ids.add(ligand_id)
    
    ligand_list = list(unique_ligand_ids)
    print(f"Processing {len(ligand_list)} unique ligands (from {len(sample_ids)} samples) with {n_processes} processes...")
    
    # Step 2: Filter unique ligands in parallel
    ligand_keep_map = {}
    
    with Pool(
        n_processes,
        initializer=init_worker,
        initargs=(ref_fingerprints, smiles_dict, similarity_threshold)
    ) as pool:
        results = list(tqdm(
            pool.imap(filter_single_ligand_worker, ligand_list),
            total=len(ligand_list),
            desc="Filtering ligands"
        ))
    
    # Convert ligand results to mapping
    for ligand_id, keep in results:
        ligand_keep_map[ligand_id] = keep
    
    # Step 3: Map back to sample IDs
    sample_keep_map = {}
    for sample_id in sample_ids:
        if '_' in sample_id:
            ligand_id = sample_id.split('_')[1]
            sample_keep_map[sample_id] = ligand_keep_map.get(ligand_id, False)
        else:
            sample_keep_map[sample_id] = False  # Invalid format
    
    # Statistics
    total_ligands = len(ligand_keep_map)
    kept_ligands = sum(ligand_keep_map.values())
    removed_ligands = total_ligands - kept_ligands
    
    total_samples = len(sample_keep_map)
    kept_samples = sum(sample_keep_map.values())
    removed_samples = total_samples - kept_samples
    
    print(f"Ligand filtering results:")
    print(f"  Total unique ligands: {total_ligands}")
    print(f"  Kept ligands: {kept_ligands}")
    print(f"  Removed ligands: {removed_ligands}")
    print(f"  Ligand retention rate: {kept_ligands/total_ligands*100:.2f}%")
    
    print(f"Sample filtering results:")
    print(f"  Total samples: {total_samples}")
    print(f"  Kept samples: {kept_samples}")
    print(f"  Removed samples: {removed_samples}")
    print(f"  Sample retention rate: {kept_samples/total_samples*100:.2f}%")
    
    return sample_keep_map

def process_single_dataset(id_json_path, smiles_json_path, ref_fingerprints, similarity_threshold=0.4, n_processes=None):
    """Process a single dataset with ID list and SMILES mapping"""
    if n_processes is None:
        n_processes = min(cpu_count(), 8)  # Limit to 8 processes to avoid memory issues
    elif n_processes == -1:
        n_processes = cpu_count()  # Use all available cores

    print(f"\nProcessing dataset: {id_json_path}")
    
    # Load ID list
    with open(id_json_path, 'r') as f:
        id_list = json.load(f)
    print(f"Loaded {len(id_list)} IDs from {id_json_path}")
    
    # Load SMILES mapping
    with open(smiles_json_path, 'r') as f:
        smiles_dict = json.load(f)
    print(f"Loaded {len(smiles_dict)} SMILES mappings from {smiles_json_path}")
    
    # Filter IDs based on Tanimoto similarity
    sample_keep_map = filter_unique_ligands_parallel(
        set(id_list), ref_fingerprints, smiles_dict, 
        similarity_threshold, n_processes
    )
    
    # Get filtered IDs
    filtered_ids = [lid for lid in id_list if sample_keep_map.get(lid, False)]
    
    # Save results
    output_path = Path(id_json_path).parent / "id.json"
    with open(output_path, 'w') as f:
        json.dump(filtered_ids, f, indent=2)
    
    # Print statistics
    original_count = len(id_list)
    filtered_count = len(filtered_ids)
    removed_count = original_count - filtered_count
    retention_rate = filtered_count / original_count * 100 if original_count > 0 else 0
    
    print(f"✔ Results saved to: {output_path}")
    print(f"✔ Original IDs: {original_count}")
    print(f"✔ Filtered IDs: {filtered_count}")
    print(f"✔ Removed IDs: {removed_count}")
    print(f"✔ Retention rate: {retention_rate:.2f}%")
    
    return filtered_ids, original_count, filtered_count


def filter_multiple_datasets(dataset_list, similarity_threshold=0.4, n_processes=None):
    """Filter multiple datasets based on Tanimoto similarity to reference compounds
    
    Args:
        dataset_list: List of tuples, each containing (id_json_path, smiles_json_path)
        similarity_threshold: Tanimoto similarity threshold for filtering (default: 0.4)
        n_processes: Number of processes for parallel processing (default: auto)
    """
    print("Starting Tanimoto similarity filtering process...")
    print(f"Similarity threshold: {similarity_threshold}")
    print(f"Number of datasets to process: {len(dataset_list)}")
    
    # Check if reference files are provided
    if not REFERENCE_SMILES_FILES:
        print("Error: REFERENCE_SMILES_FILES is empty. Please add reference file paths.")
        return
    
    # Load reference SMILES and compute fingerprints (once for all datasets)
    reference_smiles = load_reference_smiles(REFERENCE_SMILES_FILES)
    if not reference_smiles:
        print("Error: No reference SMILES loaded.")
        return
    
    print("Computing fingerprints for reference compounds...")
    ref_fingerprints, valid_ref_smiles = compute_fingerprints(list(reference_smiles))
    print(f"Successfully computed {len(ref_fingerprints)} reference fingerprints")
    
    # Process each dataset
    total_original_ids = 0
    total_filtered_ids = 0
    
    for i, (id_json_path, smiles_json_path) in enumerate(dataset_list, 1):
        print(f"\n{'='*60}")
        print(f"Processing dataset {i}/{len(dataset_list)}")
        print(f"{'='*60}")
        
        try:
            filtered_ids, original_count, filtered_count = process_single_dataset(
                id_json_path, smiles_json_path, ref_fingerprints, 
                similarity_threshold, n_processes
            )
            total_original_ids += original_count
            total_filtered_ids += filtered_count
            
        except Exception as e:
            print(f"Error processing dataset {id_json_path}: {e}")
            continue
    
    # Print overall statistics
    print(f"\n{'='*60}")
    print(f"OVERALL STATISTICS")
    print(f"{'='*60}")
    print(f"Total datasets processed: {len(dataset_list)}")
    print(f"Total original IDs: {total_original_ids}")
    print(f"Total filtered IDs: {total_filtered_ids}")
    print(f"Overall retention rate: {total_filtered_ids/total_original_ids*100:.2f}%" if total_original_ids > 0 else "N/A")
    
    print("\n🎉 Multi-dataset Tanimoto similarity filtering completed successfully!")

def main():
    """Main function - example usage of multi-dataset filtering"""
    # Example dataset list - replace with your actual dataset paths
    dataset_list = [
        # ("./data/pubchem/id.json", "./data/pubchem/smiles.json"),
        ("./data/sair/id.json", "./data/sair/smiles.json"),
    ]
    
    if not dataset_list:
        print("Please modify the dataset_list in main() function with your actual dataset paths.")
        print("Example usage:")
        print("dataset_list = [")
        print('    ("/path/to/dataset1/id.json", "/path/to/dataset1/smiles.json"),')
        print('    ("/path/to/dataset2/id.json", "/path/to/dataset2/smiles.json"),')
        print("]")
        return
    
    # Filter multiple datasets
    filter_multiple_datasets(
        dataset_list=dataset_list,
        similarity_threshold=0.4,
        n_processes=-1
    )

if __name__ == "__main__":
    main()
