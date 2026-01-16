import subprocess
import tempfile
import json
from pathlib import Path
from typing import Set, Dict, List
import pandas as pd
from tqdm import tqdm

from affinity.data.utils import split_field, hash_sequence

################################################################################
# Filter processed data to exclude test set related proteins
# 1. Cluster proteins using mmseqs2
# 2. Collect test set protein hashes and cluster to find excluded hashes
# 3. Filter processed data to exclude test set related proteins
# 4. Save filtered data
################################################################################

def cluster_proteins_mmseqs(hash_to_seq: Dict[str, str], min_seq_id: float = 0.9, coverage: float = 0.01) -> Dict[str, List[str]]:
    """Cluster proteins using mmseqs2"""
    if not hash_to_seq:
        return {}
    
    print(f"Start clustering {len(hash_to_seq)} proteins...")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        input_fasta = tmp_path / "input.fasta"
        with open(input_fasta, 'w') as f:
            for seq_hash, seq in hash_to_seq.items():
                if seq.strip():
                    f.write(f">{seq_hash}\n{seq}\n")
        
        try:
            subprocess.run([
                "mmseqs", "easy-cluster",
                str(input_fasta),
                str(tmp_path / "cluster"),
                str(tmp_path),
                "--min-seq-id", str(min_seq_id),
                "--cov-mode", "0",
                "-c", str(coverage),
                "--threads", "4"
            ], check=True, capture_output=True)
            
            cluster_file = tmp_path / "cluster_cluster.tsv"
            if not cluster_file.exists():
                print(f"Warning: cluster result file not found: {cluster_file}")
                return {}
            
            clusters = {}
            with open(cluster_file, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        representative = parts[0]
                        member = parts[1]
                        if representative not in clusters:
                            clusters[representative] = []
                        clusters[representative].append(member)
            
            print(f"Clustering completed, {len(clusters)} clusters generated")
            return clusters
            
        except subprocess.CalledProcessError as e:
            print(f"mmseqs clustering failed: {e}")
            print(f"Standard error output: {e.stderr.decode() if e.stderr else 'None'}")
            return {}
        except FileNotFoundError:
            print("Error: mmseqs command not found. Please ensure mmseqs is correctly installed and in PATH.")
            return {}

def collect_test_sequences(test_prots_paths: List[Path]) -> Set[str]:
    """Collect all unique sequences from test sets"""
    print("Collecting test set sequences...")
    
    test_sequences = set()
    total_test_proteins = 0
    
    for test_prots_path in test_prots_paths:
        if test_prots_path.exists():
            print(f"Loading test set from: {test_prots_path}")
            with open(test_prots_path, 'r') as f:
                test_prots = json.load(f)
            
            # Only collect sequences, ignore protein IDs to avoid conflicts
            sequences = set(seq.strip() for seq in test_prots.values() if seq.strip())
            test_sequences.update(sequences)
            total_test_proteins += len(test_prots)
            print(f"Loaded {len(test_prots)} proteins ({len(sequences)} unique sequences) from {test_prots_path}")
        else:
            print(f"Warning: test set file not found: {test_prots_path}")
    
    print(f"Total test proteins: {total_test_proteins}, unique sequences: {len(test_sequences)}")
    return test_sequences

def filter_single_training_dataset(test_sequences: Set[str], id_path: Path, prots_path: Path) -> None:
    """Filter a single training dataset based on test sequences"""
    print(f"\nProcessing training dataset: {id_path}")
    
    if not prots_path.exists():
        print(f"Warning: training prots file not found: {prots_path}")
        return
    
    if not id_path.exists():
        print(f"Warning: training ID file not found: {id_path}")
        return
    
    # Load current training dataset
    with open(prots_path, 'r') as f:
        train_prot_to_seq = json.load(f)
    
    print(f"Current training dataset contains {len(train_prot_to_seq)} proteins")
    
    # Create hash-to-sequence mapping for clustering
    all_sequences = test_sequences | set(seq.strip() for seq in train_prot_to_seq.values() if seq.strip())
    all_hash_to_seq = {}
    
    for seq in all_sequences:
        if seq.strip():
            seq_hash = hash_sequence(seq.strip())
            all_hash_to_seq[seq_hash] = seq.strip()
    
    print(f"Clustering {len(all_hash_to_seq)} unique sequences...")
    
    # Cluster sequences
    clusters = cluster_proteins_mmseqs(all_hash_to_seq)
    
    exclude_prot_ids = set()
    
    if not clusters:
        print("Clustering failed, using exact sequence matching")
        # Fall back to exact sequence matching
        for prot_id, seq in train_prot_to_seq.items():
            if seq.strip() in test_sequences:
                exclude_prot_ids.add(prot_id)
    else:
        # Find excluded protein IDs based on clustering
        test_hashes = {hash_sequence(seq) for seq in test_sequences}
        
        # First pass: collect all hashes that need to be excluded
        exclude_hashes = set()
        excluded_clusters = 0
        for representative, members in clusters.items():
            cluster_hashes = set(members)
            if cluster_hashes & test_hashes:
                exclude_hashes.update(cluster_hashes)
                excluded_clusters += 1
        
        print(f"Found {excluded_clusters} clusters containing test sequences")
        
        # Second pass: find training protein IDs with excluded sequences
        for prot_id, seq in train_prot_to_seq.items():
            seq_hash = hash_sequence(seq.strip())
            if seq_hash in exclude_hashes:
                exclude_prot_ids.add(prot_id)
    
    print(f"Found {len(exclude_prot_ids)} proteins to exclude from current training dataset")
    
    # Load and filter ID list
    with open(id_path, 'r') as f:
        id_list = json.load(f)
    
    original_count = len(id_list)
    print(f"Original IDs: {original_count}")
    
    # Filter IDs
    filtered_ids = []
    excluded_count = 0
    
    for compound_id in tqdm(id_list, desc="Filtering IDs", leave=False):
        # Extract protein ID from compound_id (format: {prot_id}_{ligand_id})
        if '_' in compound_id:
            prot_id = compound_id.split('_')[0]
            if prot_id in exclude_prot_ids:
                excluded_count += 1
                continue
        filtered_ids.append(compound_id)
    
    filtered_count = len(filtered_ids)
    
    # Save filtered IDs
    filtered_id_path = id_path.parent / "id.json"
    with open(filtered_id_path, 'w') as f:
        json.dump(filtered_ids, f, indent=2)
    
    print(f"✔ Filtered IDs saved to {filtered_id_path}")
    print(f"✔ Original IDs: {original_count}")
    print(f"✔ Excluded IDs: {excluded_count}")
    print(f"✔ Final IDs: {filtered_count}")
    print(f"✔ Retention rate: {filtered_count/original_count*100:.2f}%")

def filter_training_datasets(test_prots_paths: List[Path], train_datasets: List[tuple]):
    """Filter training datasets to exclude test set related proteins"""
    
    # Collect all test sequences (avoid ID conflicts by only using sequences)
    test_sequences = collect_test_sequences(test_prots_paths)
    
    if not test_sequences:
        print("No test sequences found, skipping filtering...")
        return
    
    # Process each training dataset separately
    print(f"\nProcessing {len(train_datasets)} training datasets...")
    for id_path, prots_path in tqdm(train_datasets, desc="Processing training datasets"):
        filter_single_training_dataset(test_sequences, id_path, prots_path)

if __name__ == "__main__":
    # Example usage - modify these paths according to your actual data structure
    test_prots_paths = [
        Path("./data/casp16/prots.json"),
        Path("./data/openfe/prots.json"),
        Path("./data/fep4/prots.json"),
        Path("./data/sair/val_prots.json"),
        # Path("./data/mf-pcba/prot.json"),
        # Path("./data/lit-pcba/prots.json"),
        # Add more test sets as needed
    ]
    
    train_datasets = [
        # (Path("./data/pubchem/all_id.json"), Path("./data/pubchem/prots.json")),
        (Path("./data/sair/id.json"), Path("./data/sair/prots.json")),
        # Add more training datasets as needed
    ]
    
    filter_training_datasets(test_prots_paths, train_datasets)