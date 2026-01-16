#############################################
# Use ESM3 to generate protein representations and save to LMDB.
#############################################

import os
# Set this environment variable to potentially help with GPU memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
import torch
import json
import lmdb
import pickle
import gc
from pathlib import Path
from typing import Optional, Dict
from collections import defaultdict
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

try:
    from esm.models.esm3 import ESM3
    from esm.sdk.api import ESM3InferenceClient, LogitsConfig, ESMProtein
except ImportError:
    raise ImportError("ESM library not found. Please install it with: pip install esm")

class ProteinRepresentationGenerator:
    """Generate protein representations using the ESM3 model from a sequence dictionary."""
    
    def __init__(self, model_name: str = "esm3_sm_open_v1", device: str = "auto"):
        """
        Initialize the protein representation generator.
        
        Args:
            model_name (str): The ESM3 model name to use.
            device (str): The device to use ('auto', 'cpu', 'cuda').
        """
        self.model_name = model_name
        self.device = self._get_device(device)
        self.model = self._load_model()
        
    def _get_device(self, device: str) -> torch.device:
        """Determine the device to use."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                print("Warning: CUDA not available, using CPU. ESM3 models are optimized for GPU.")
                return torch.device("cpu")
        return torch.device(device)
    
    def _load_model(self):
        """Load the ESM3 model."""
        print(f"Loading ESM3 model: {self.model_name}")
        model: ESM3InferenceClient = ESM3.from_pretrained(self.model_name).to(self.device)
        model.eval()
        return model
    
    def generate_representation(self, sequence: str) -> Optional[torch.Tensor]:
        """
        Generate a protein representation for a single sequence using ESM3.
        
        Args:
            sequence (str): The protein amino acid sequence.
            
        Returns:
            torch.Tensor: A representation tensor of shape [L, D] or None if generation fails.
        """
        if not sequence:
            print("Warning: Received an empty sequence. Skipping.")
            return None
            
        try:
            protein = ESMProtein(sequence=sequence)
            input_tensor = self.model.encode(protein).to(self.device)
            output = self.model.logits(
                input_tensor, LogitsConfig(return_embeddings=True)
            )
            embeddings = output.embeddings.squeeze(0)[1:-1]
            
            if embeddings.shape[0] != len(sequence):
                raise ValueError(f"Embedding length {embeddings.shape[0]} does not match sequence length {len(sequence)}")
                
            return embeddings.cpu()
            
        except Exception as e:
            print(f"Error generating representation for a sequence of length {len(sequence)}: {e}")
            return None
    
    def process_sequences_from_json(self, json_path: str, output_lmdb_path: str, batch_size: int = 10000):
        """
        Process sequences from a JSON file and save their representations to an LMDB database.
        
        Args:
            json_path (str): Path to the input JSON file (protein_id -> sequence).
            output_lmdb_path (str): Path to the output LMDB database directory.
            batch_size (int): Number of sequences to process before writing to LMDB and clearing memory.
        """
        try:
            with open(json_path, 'r') as f:
                sequences_dict = json.load(f)
            print(f"Loaded {len(sequences_dict)} sequences from {json_path}")
        except Exception as e:
            print(f"Error loading JSON file {json_path}: {e}")
            return
            
        print(f"Using device: {self.device}")
        
        # Load existing protein IDs from the LMDB database if it exists
        existing_prot_ids = set()
        if os.path.isdir(output_lmdb_path):
            print(f"Found existing LMDB at {output_lmdb_path}. Checking for processed proteins.")
            try:
                env = lmdb.open(output_lmdb_path, readonly=True, lock=False)
                with env.begin() as txn:
                    num_entries = txn.stat()['entries']
                    cursor = txn.cursor()
                    for key_bytes in tqdm(cursor.iternext(keys=True, values=False), desc="Scanning LMDB", total=num_entries):
                        existing_prot_ids.add(key_bytes.decode('utf-8'))
                env.close()
                print(f"Loaded {len(existing_prot_ids)} existing protein IDs.")
            except lmdb.Error as e:
                print(f"Warning: Could not read existing LMDB. May be empty or corrupted. Error: {e}")

        # Determine which proteins need to be processed
        all_prot_ids = set(sequences_dict.keys())
        missing_prot_ids = all_prot_ids - existing_prot_ids
        
        if not missing_prot_ids:
            print("All proteins already processed! ✨")
            return
        
        print(f"Found {len(missing_prot_ids)} new proteins to process.")
        
        # Group missing protein IDs by their sequence to de-duplicate
        sequence_to_ids = defaultdict(list)
        for prot_id in missing_prot_ids:
            sequence = sequences_dict.get(prot_id)
            if sequence:
                sequence_to_ids[sequence].append(prot_id)
        
        unique_sequences_to_process = list(sequence_to_ids.keys())
        print(f"Processing {len(unique_sequences_to_process)} unique sequences for these new proteins.")
        print(f"Using batch size: {batch_size}")

        # Estimate map size (100MB per protein should be very safe)
        total_proteins = len(existing_prot_ids) + len(missing_prot_ids)
        map_size = total_proteins * 100 * 1024 * 1024

        # Process sequences in batches to avoid memory overflow
        total_processed = 0
        total_written = 0
        
        # Split sequences into batches
        for batch_start in range(0, len(unique_sequences_to_process), batch_size):
            batch_end = min(batch_start + batch_size, len(unique_sequences_to_process))
            batch_sequences = unique_sequences_to_process[batch_start:batch_end]
            
            print(f"\nProcessing batch {batch_start//batch_size + 1}/{(len(unique_sequences_to_process) + batch_size - 1)//batch_size}")
            print(f"Batch size: {len(batch_sequences)} sequences")
            
            # Process current batch
            batch_sequence_to_representation = {}
            with tqdm(total=len(batch_sequences), desc=f"Processing batch sequences") as pbar:
                for sequence in batch_sequences:
                    representation = self.generate_representation(sequence)
                    if representation is not None:
                        batch_sequence_to_representation[sequence] = representation
                    pbar.update(1)
                    total_processed += 1

            # Write current batch to LMDB immediately
            if batch_sequence_to_representation:
                print(f"Writing {len(batch_sequence_to_representation)} representations from current batch to LMDB")
                
                env = lmdb.open(output_lmdb_path, map_size=map_size)
                with env.begin(write=True) as txn:
                    pbar_write = tqdm(batch_sequence_to_representation.items(), desc="Writing batch to LMDB")
                    for sequence, representation in pbar_write:
                        # Serialize the tensor
                        value_bytes = pickle.dumps(representation)
                        # Write an entry for each protein ID associated with this sequence
                        for prot_id in sequence_to_ids[sequence]:
                            key_bytes = prot_id.encode('utf-8')
                            txn.put(key_bytes, value_bytes)
                            total_written += 1
                env.close()
                
                print(f"Successfully wrote batch to LMDB. Total written so far: {total_written}")
            else:
                print("No valid representations in current batch.")
            
            # Clear memory after each batch
            del batch_sequence_to_representation
            gc.collect()
            
            # Clear GPU memory if using CUDA
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            print(f"Memory cleared after batch processing.")

        if total_written > 0:
            print(f"\nSuccessfully finished processing all batches.")
            print(f"Total sequences processed: {total_processed}")
            print(f"Total protein entries written: {total_written}")
        else:
            print("No new valid representations were generated.")
        
        print(f"Total proteins in database: {len(existing_prot_ids) + len(missing_prot_ids)}")


def main():
    """Main function to run the protein representation generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate protein representations using ESM3 and save to LMDB.")
    parser.add_argument("--input_json", default="./data/mf-pcba/prots.json", 
                        help="Path to the input JSON file (prot_id -> sequence).")
    parser.add_argument("--output_lmdb", default="./data/mf-pcba/repr/esm3.lmdb", 
                        help="Path to save the output LMDB database directory.")
    parser.add_argument("--model", default="esm3_sm_open_v1", 
                        help="ESM3 model to use (e.g., 'esm3_sm_open_v1', 'esm3_md_open_v1').")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                        help="Device to use for computation.")
    parser.add_argument("--batch_size", type=int, default=10000,
                        help="Number of sequences to process before writing to LMDB and clearing memory.")

    args = parser.parse_args()
    
    # Ensure the output directory exists
    os.makedirs(Path(args.output_lmdb).parent, exist_ok=True)
    
    generator = ProteinRepresentationGenerator(
        model_name=args.model,
        device=args.device
    )
    
    generator.process_sequences_from_json(
        json_path=args.input_json,
        output_lmdb_path=args.output_lmdb,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()