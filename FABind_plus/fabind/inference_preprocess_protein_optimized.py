import torch
from tqdm import tqdm
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="max_split_size_mb:128"
import argparse
import logging
import lmdb
import pickle
from typing import Dict, List, Tuple, Any, Optional
from joblib import Parallel, delayed, cpu_count
from utils.inference_pdb_utils import extract_protein_structure, extract_esm_feature
import esm

# ============ Configuration Constants ============
LMDB_MAP_SIZE_STRUCTURE = 1000 * 1024**3  # 1TB
LMDB_MAP_SIZE_ESM = 1000 * 1024**3        # 1TB
DEFAULT_STRUCTURE_BATCH_SIZE = 50000
DEFAULT_ESM_BATCH_SIZE = 8  # GPU batch size for ESM inference
DEFAULT_ESM_CHECKPOINT_FREQ = 10000  # Save to LMDB every N proteins

# ============ Logging Setup ============
def setup_logger():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

# ============ ESM Batch Processor ============
class ESMBatchProcessor:
    """ESM feature extractor with batch processing support"""
    
    def __init__(self, device: Optional[str] = None):
        """Initialize ESM model (load once and reuse)
        
        Args:
            device: Device to run model on ('cuda' or 'cpu')
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logging.info(f"Loading ESM-2 model on {self.device}...")
        # Load ESM-2 model (same as original)
        self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.model.to(self.device)
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model.eval()  # Disable dropout for deterministic results
        logging.info("ESM-2 model loaded successfully!")
    
    def extract_batch(self, sequences_dict: Dict[str, str]) -> Dict[str, torch.Tensor]:
        """Extract ESM features for a batch of sequences
        
        Args:
            sequences_dict: Dictionary of {pdb_id: sequence_string}
        
        Returns:
            Dictionary of {pdb_id: esm_features} where features shape is [seq_len, dim]
        """
        if not sequences_dict:
            return {}
        
        # Prepare batch data and record original sequence lengths
        data = []
        pdb_ids = []
        seq_lengths = []
        
        for pdb_id, seq in sequences_dict.items():
            data.append((pdb_id, seq))
            pdb_ids.append(pdb_id)
            seq_lengths.append(len(seq))
        
        # Convert to ESM batch format (will pad to max length)
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)
        
        # Batch inference
        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[33])
        token_representations = results["representations"][33]  # [batch_size, padded_len+2, dim]
        
        # Extract features for each protein, truncate to original length
        esm_features = {}
        for i, pdb_id in enumerate(pdb_ids):
            # Remove start token (index 0) and extract only original sequence length
            # token_representations[i]: [padded_len+2, dim]
            # [1 : seq_lengths[i] + 1]: skip start token, take original length
            features = token_representations[i][1 : seq_lengths[i] + 1]
            
            # Verify length matches original sequence
            assert features.shape[0] == seq_lengths[i], \
                f"Length mismatch for {pdb_id}: expected {seq_lengths[i]}, got {features.shape[0]}"
            
            # Move to CPU to free GPU memory (important for accumulation!)
            esm_features[pdb_id] = features.cpu()
        
        return esm_features

# ============ LMDB Helper Functions ============
class LMDBHelper:
    """Helper class for LMDB operations"""
    
    @staticmethod
    def append_items(db_path: str, items_dict: Dict, prefix: str, 
                     failed_list: Optional[List] = None, map_size: int = LMDB_MAP_SIZE_STRUCTURE,
                     show_progress: bool = True):
        """Append items to LMDB (incremental save)"""
        env = lmdb.open(db_path, map_size=map_size)
        with env.begin(write=True) as txn:
            # Save items with progress bar
            items_iter = tqdm(items_dict.items(), desc=f"Saving {prefix}*", disable=not show_progress) if show_progress else items_dict.items()
            for item_id, data in items_iter:
                key = f"{prefix}{item_id}".encode()
                value = pickle.dumps(data)
                txn.put(key, value)
            
            # Update failed list if provided
            if failed_list is not None:
                failed_key = b"failed_list"
                # Load existing failed list
                existing_failed = txn.get(failed_key)
                if existing_failed:
                    all_failed = pickle.loads(existing_failed)
                    all_failed.extend(failed_list)
                else:
                    all_failed = failed_list
                txn.put(failed_key, pickle.dumps(all_failed))
        env.close()
    
    @staticmethod
    def get_all_ids_by_prefix(db_path: str, prefix: str) -> List[str]:
        """Get all IDs from LMDB by prefix (only scan keys, not values)"""
        if not os.path.exists(db_path):
            return []
        
        ids = []
        try:
            env = lmdb.open(db_path, readonly=True)
            with env.begin() as txn:
                cursor = txn.cursor()
                prefix_bytes = prefix.encode()
                # Efficiently scan by prefix, only get keys
                if cursor.set_range(prefix_bytes):
                    for key in cursor.iternext(keys=True, values=False):
                        key_str = key.decode()
                        if key_str.startswith(prefix):
                            item_id = key_str[len(prefix):]
                            ids.append(item_id)
                        else:
                            break
            env.close()
        except Exception as e:
            logging.warning(f"Failed to get IDs from LMDB: {e}")
            return []
        
        return ids
    
    @staticmethod
    def load_items_by_keys(db_path: str, keys: List[str]) -> Dict:
        """Load items from LMDB by keys"""
        results = {}
        try:
            env = lmdb.open(db_path, readonly=True)
            with env.begin() as txn:
                for key in keys:
                    value = txn.get(key.encode())
                    if value:
                        results[key] = pickle.loads(value)
            env.close()
        except Exception as e:
            logging.warning(f"Failed to load items from LMDB: {e}")
            return {}
        
        return results
    
    @staticmethod
    def load_all_by_prefix(db_path: str, prefix: str) -> Dict:
        """Load all items from LMDB by prefix"""
        if not os.path.exists(db_path):
            return {}
        
        results = {}
        try:
            env = lmdb.open(db_path, readonly=True)
            with env.begin() as txn:
                cursor = txn.cursor()
                prefix_bytes = prefix.encode()
                if cursor.set_range(prefix_bytes):
                    for key, value in tqdm(cursor, desc=f"Loading {prefix}*"):
                        key_str = key.decode()
                        if key_str.startswith(prefix):
                            item_id = key_str[len(prefix):]
                            results[item_id] = pickle.loads(value)
                        else:
                            break
            env.close()
        except Exception as e:
            logging.warning(f"Failed to load items from LMDB: {e}")
            return {}
        
        return results
    
    @staticmethod
    def get_failed_list(db_path: str) -> List:
        """Get failed list from LMDB"""
        if not os.path.exists(db_path):
            return []
        
        try:
            env = lmdb.open(db_path, readonly=True)
            with env.begin() as txn:
                failed_value = txn.get(b"failed_list")
                if failed_value:
                    failed_list = pickle.loads(failed_value)
                else:
                    failed_list = []
            env.close()
            return failed_list
        except Exception as e:
            logging.warning(f"Failed to get failed list: {e}")
            return []

# ============ Structure Extraction ============
def extract_single_structure(pdb_file_info: Tuple[str, str]) -> Tuple[str, Any, bool, Optional[str]]:
    """Extract structure for a single PDB file
    
    Args:
        pdb_file_info: Tuple of (pdb_file, pdb_file_dir)
    
    Returns:
        Tuple of (pdb_id, structure, success, error_message)
    """
    pdb_file, pdb_file_dir = pdb_file_info
    try:
        pdb = pdb_file.split(".")[0]
        pdb_filepath = os.path.join(pdb_file_dir, pdb_file)
        protein_structure = extract_protein_structure(pdb_filepath)
        protein_structure['name'] = pdb
        return pdb, protein_structure, True, None
    except Exception as e:
        return pdb_file.split(".")[0], None, False, str(e)

# ============ Phase 1: Structure Extraction ============
def process_structures_phase(args, structure_checkpoint: str) -> Dict:
    """Phase 1: Extract protein structures in parallel
    
    Args:
        args: Command line arguments
        structure_checkpoint: Path to structure LMDB checkpoint
    
    Returns:
        Dictionary with statistics (total, success, failed)
    """
    logging.info(f"=== Phase 1: Extracting protein structures (using {args.num_threads} threads) ===")
    
    # Get list of PDB files
    pdb_files = [f for f in os.listdir(args.pdb_file_dir) if f.endswith('.pdb')]
    logging.info(f"Found {len(pdb_files)} PDB files")
    
    if not pdb_files:
        return {'total': 0, 'success': 0, 'failed': 0}
    
    # Check for existing processed structures (resume functionality)
    processed_protein_ids = set(LMDBHelper.get_all_ids_by_prefix(structure_checkpoint, 'protein_'))
    if processed_protein_ids:
        logging.info(f"Found {len(processed_protein_ids)} already processed structures")
        # Filter out already processed files
        remaining_files = [f for f in pdb_files if f.split('.')[0] not in processed_protein_ids]
        logging.info(f"Remaining to process: {len(remaining_files)}")
    else:
        remaining_files = pdb_files
        logging.info(f"No existing checkpoint found, processing all files")
    
    if not remaining_files:
        logging.info("All structures already processed, skipping Phase 1")
        return {'total': len(pdb_files), 'success': len(processed_protein_ids), 'failed': 0}
    
    # Calculate batch size
    if args.batch_size == 0:
        batch_size = min(args.num_threads * 1000, DEFAULT_STRUCTURE_BATCH_SIZE)
    else:
        batch_size = args.batch_size
    
    total_batches = (len(remaining_files) + batch_size - 1) // batch_size
    logging.info(f"Processing {len(remaining_files)} files in {total_batches} batches of {batch_size}")
    
    total_success = 0
    total_failed = 0
    
    # Process in batches
    for batch_idx in range(0, len(remaining_files), batch_size):
        batch_end = min(batch_idx + batch_size, len(remaining_files))
        batch_files = remaining_files[batch_idx:batch_end]
        batch_infos = [(pdb_file, args.pdb_file_dir) for pdb_file in batch_files]
        batch_num = batch_idx // batch_size + 1
        
        logging.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_files)} files)")
        
        # Process current batch in parallel
        batch_results = Parallel(n_jobs=args.num_threads, backend='loky')(
            delayed(extract_single_structure)(info) 
            for info in tqdm(batch_infos, desc=f"Batch {batch_num}/{total_batches}")
        )
        
        # Separate success and failed results, extract sequences in one pass
        batch_protein_dict = {}
        batch_seq_dict = {}
        batch_failed_list = []
        
        for pdb, structure, success, error in batch_results:
            if success:
                batch_protein_dict[pdb] = structure
                batch_seq_dict[pdb] = structure['seq']  # Extract sequence simultaneously
                total_success += 1
            else:
                batch_failed_list.append({'pdb': pdb, 'error': error})
                total_failed += 1
                logging.warning(f"Failed to process {pdb}: {error}")
        
        # Save both structures and sequences to LMDB (batch write is faster than one-by-one)
        LMDBHelper.append_items(
            structure_checkpoint, 
            batch_protein_dict, 
            prefix='protein_',
            failed_list=batch_failed_list,
            map_size=LMDB_MAP_SIZE_STRUCTURE,
            show_progress=True
        )
        
        # Save sequences separately for efficient ESM processing
        LMDBHelper.append_items(
            structure_checkpoint,
            batch_seq_dict,
            prefix='seq_',
            failed_list=None,
            map_size=LMDB_MAP_SIZE_STRUCTURE,
            show_progress=False  # Don't show progress for sequences (fast)
        )
        
        logging.info(f"Batch {batch_num} completed: {len(batch_protein_dict)} success, {len(batch_failed_list)} failed")
        
        # Clear memory immediately
        batch_protein_dict.clear()
        batch_seq_dict.clear()
        batch_failed_list.clear()
    
    logging.info(f"Structure extraction completed: {total_success} success, {total_failed} failed")
    
    return {
        'total': len(pdb_files),
        'success': total_success,
        'failed': total_failed
    }

# ============ Phase 2: ESM Feature Extraction ============
def process_esm_phase(structure_checkpoint: str, esm_checkpoint: str, 
                     esm_batch_size: int = DEFAULT_ESM_BATCH_SIZE,
                     checkpoint_freq: int = DEFAULT_ESM_CHECKPOINT_FREQ) -> Dict:
    """Phase 2: Extract ESM features with batch processing
    
    Args:
        structure_checkpoint: Path to structure LMDB checkpoint
        esm_checkpoint: Path to ESM LMDB checkpoint
        esm_batch_size: Batch size for ESM inference (e.g., 32)
        checkpoint_freq: Save to LMDB every N proteins (e.g., 10000)
    
    Returns:
        Dictionary with statistics (total, success, failed)
    """
    logging.info("=== Phase 2: Extracting ESM features ===")
    
    # Load all sequences to memory (sequences are small, only strings)
    # Do this in one LMDB scan instead of separate ID scan + data scan
    logging.info("Loading all sequences to memory...")
    all_sequences = LMDBHelper.load_all_by_prefix(structure_checkpoint, 'seq_')
    logging.info(f"Total proteins available for ESM: {len(all_sequences)}")
    
    if not all_sequences:
        return {'total': 0, 'success': 0, 'failed': 0}
    
    # Check for existing processed ESM features (resume functionality)
    # Only scan keys, don't load ESM data
    processed_esm_ids = set(LMDBHelper.get_all_ids_by_prefix(esm_checkpoint, 'esm_'))
    if processed_esm_ids:
        logging.info(f"Found {len(processed_esm_ids)} already processed ESM features")
        # Filter sequences by processed ESM IDs
        remaining_sequences = {sid: seq for sid, seq in all_sequences.items() 
                              if sid not in processed_esm_ids}
        logging.info(f"Remaining to process: {len(remaining_sequences)}")
    else:
        remaining_sequences = all_sequences
        logging.info(f"No existing ESM checkpoint found, processing all sequences")
    
    if not remaining_sequences:
        logging.info("All ESM features already processed, skipping Phase 2")
        return {'total': len(all_sequences), 'success': len(processed_esm_ids), 'failed': 0}
    
    logging.info(f"Sequences loaded to memory: {len(remaining_sequences)}")
    
    # Initialize ESM batch processor (load model once)
    esm_processor = ESMBatchProcessor()
    
    # Process ESM in small batches for inference, accumulate for checkpoint saving
    remaining_seq_ids = list(remaining_sequences.keys())
    total_batches = (len(remaining_seq_ids) + esm_batch_size - 1) // esm_batch_size
    logging.info(f"Processing ESM features: inference batch_size={esm_batch_size}, checkpoint every {checkpoint_freq} proteins")
    logging.info(f"Total inference batches: {total_batches}")
    
    total_success = 0
    total_failed = 0
    
    # Accumulate results before saving to LMDB
    accumulated_esm_dict = {}
    accumulated_failed_list = []
    
    for batch_idx in tqdm(range(0, len(remaining_seq_ids), esm_batch_size), desc="Processing ESM features"):
        batch_end = min(batch_idx + esm_batch_size, len(remaining_seq_ids))
        batch_pdb_ids = remaining_seq_ids[batch_idx:batch_end]
        batch_num = batch_idx // esm_batch_size + 1
        
        # Prepare batch sequences (already in memory)
        batch_seqs = {pdb_id: remaining_sequences[pdb_id] for pdb_id in batch_pdb_ids}
        
        # Batch inference (small batch for GPU efficiency)
        try:
            batch_esm_dict = esm_processor.extract_batch(batch_seqs)
            accumulated_esm_dict.update(batch_esm_dict)
            total_success += len(batch_esm_dict)
        except Exception as e:
            # If batch fails, try one by one
            logging.warning(f"Batch {batch_num} failed: {e}, trying individual processing...")
            
            for pdb_id, seq in batch_seqs.items():
                try:
                    individual_result = esm_processor.extract_batch({pdb_id: seq})
                    accumulated_esm_dict.update(individual_result)
                    total_success += 1
                except Exception as individual_e:
                    accumulated_failed_list.append({'pdb': pdb_id, 'error': str(individual_e)})
                    total_failed += 1
                    logging.warning(f"Failed to process ESM for {pdb_id}: {individual_e}")
        
        # Save to LMDB when accumulated enough results (separate from inference batch size!)
        if len(accumulated_esm_dict) >= checkpoint_freq:
            logging.info(f"Saving checkpoint: {len(accumulated_esm_dict)} ESM features...")
            LMDBHelper.append_items(
                esm_checkpoint,
                accumulated_esm_dict,
                prefix='esm_',
                failed_list=accumulated_failed_list if accumulated_failed_list else None,
                map_size=LMDB_MAP_SIZE_ESM,
                show_progress=True
            )
            logging.info(f"Checkpoint saved at batch {batch_num}/{total_batches}")
            
            # Clear accumulated data after saving
            accumulated_esm_dict.clear()
            accumulated_failed_list.clear()
        
        # Progress logging every 100 batches
        if batch_num % 100 == 0:
            logging.info(f"Progress: {batch_num}/{total_batches} batches, {total_success} success, {total_failed} failed")
        
        # Clear GPU cache periodically
        if batch_num % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Save remaining accumulated results
    if accumulated_esm_dict:
        logging.info(f"Saving final checkpoint: {len(accumulated_esm_dict)} ESM features...")
        LMDBHelper.append_items(
            esm_checkpoint,
            accumulated_esm_dict,
            prefix='esm_',
            failed_list=accumulated_failed_list if accumulated_failed_list else None,
            map_size=LMDB_MAP_SIZE_ESM,
            show_progress=True
        )
        accumulated_esm_dict.clear()
        accumulated_failed_list.clear()
    
    logging.info(f"ESM extraction completed: {total_success} success, {total_failed} failed")
    
    # Store total count before clearing
    total_count = len(all_sequences)
    
    # Clear sequences from memory
    all_sequences.clear()
    remaining_sequences.clear()
    
    return {
        'total': total_count,
        'success': total_success,
        'failed': total_failed
    }

# ============ Phase 3: Build Final Output ============
def build_final_output(structure_checkpoint: str, esm_checkpoint: str, output_path: str, batch_size: int = 15000) -> Dict:
    """
    Phase 3: Build final output LMDB with streaming and batch commits (memory and speed efficient)

    Args:
        structure_checkpoint: Path to structure LMDB checkpoint
        esm_checkpoint: Path to ESM LMDB checkpoint
        output_path: Path to save final output LMDB
        batch_size: Number of records to write per transaction
    
    Returns:
        Dictionary with statistics
    """
    logging.info("=== Phase 3: Building final output (optimized streaming mode) ===")

    # Open all LMDB databases
    esm_env = lmdb.open(esm_checkpoint, readonly=True, lock=False)
    structure_env = lmdb.open(structure_checkpoint, readonly=True, lock=False)
    final_env = lmdb.open(output_path, map_size=LMDB_MAP_SIZE_STRUCTURE, writemap=True, sync=False)

    count = 0
    
    with esm_env.begin() as esm_txn, structure_env.begin() as structure_txn:
        total_items = esm_txn.stat()['entries']
        
        final_txn = final_env.begin(write=True)
        
        esm_cursor = esm_txn.cursor()
        
        if not esm_cursor.set_range(b'esm_'):
            logging.warning("No keys with 'esm_' prefix found.")
            pbar = []
        else:
            pbar = tqdm(esm_cursor, total=total_items, desc="Building final output")
            pbar.update(esm_cursor.key() != b'esm_')

        for esm_key, esm_value in pbar:
            esm_key_str = esm_key.decode()
            
            if not esm_key_str.startswith('esm_'):
                continue

            pdb_id = esm_key_str[4:]
            
            protein_key = f'protein_{pdb_id}'.encode()
            protein_value = structure_txn.get(protein_key)
            
            if protein_value:
                esm_feature = pickle.loads(esm_value)
                protein_structure = pickle.loads(protein_value)
                
                combined = (esm_feature, protein_structure)
                final_txn.put(pdb_id.encode(), pickle.dumps(combined))
                count += 1

                if count % batch_size == 0:
                    final_txn.commit()
                    final_txn = final_env.begin(write=True)
                    pbar.set_postfix({'processed': count})

    final_txn.commit()

    esm_env.close()
    structure_env.close()
    
    final_env.sync(True)
    final_env.close()
    
    logging.info(f"Final results saved to {output_path}")
    logging.info(f"Successfully processed {count} complete proteins")
    
    return {'total_proteins': count}

# ============ Main Function ============
def main():
    """Main function - orchestrates the three processing phases"""
    parser = argparse.ArgumentParser(description='Preprocess protein (optimized version).')
    parser.add_argument("--pdb_file_dir", type=str, default="../inference_examples/pdb_files",
                        help="Specify the pdb data path.")
    parser.add_argument("--save_pt_dir", type=str, default="../inference_examples",
                        help="Specify where to save the processed pt.")
    parser.add_argument("--num_threads", type=int, default=0,
                        help="Number of parallel jobs for structure extraction (0 = auto)")
    parser.add_argument("--batch_size", type=int, default=0,
                        help="Batch size for structure processing (0 = auto: num_threads * 1000, max 50000)")
    parser.add_argument("--esm_batch_size", type=int, default=DEFAULT_ESM_BATCH_SIZE,
                        help="Batch size for ESM inference (default: 32)")
    parser.add_argument("--esm_checkpoint_freq", type=int, default=DEFAULT_ESM_CHECKPOINT_FREQ,
                        help="Save ESM checkpoint every N proteins (default: 10000)")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logger()
    
    # Create output directory
    os.makedirs(args.save_pt_dir, exist_ok=True)
    
    # Set number of threads
    if args.num_threads == 0:
        args.num_threads = cpu_count()
    
    # Define checkpoint file paths
    structure_checkpoint = os.path.join(args.save_pt_dir, 'structure_checkpoint.lmdb')
    esm_checkpoint = os.path.join(args.save_pt_dir, 'esm_checkpoint.lmdb')
    final_output = os.path.join(args.save_pt_dir, 'processed_protein.lmdb')
    
    # Phase 1: Extract protein structures
    stats_structure = process_structures_phase(args, structure_checkpoint)
    
    # Phase 2: Extract ESM features
    stats_esm = process_esm_phase(structure_checkpoint, esm_checkpoint, args.esm_batch_size, args.esm_checkpoint_freq)
    
    # Phase 3: Build final output
    stats_final = build_final_output(structure_checkpoint, esm_checkpoint, final_output)
    
    # Print summary
    logging.info("\n=== Summary ===")
    # logging.info(f"Total PDB files: {stats_structure['total']}")
    # logging.info(f"Structure extraction - Success: {stats_structure['success']}, Failed: {stats_structure['failed']}")
    # logging.info(f"ESM extraction - Success: {stats_esm['success']}, Failed: {stats_esm['failed']}")
    logging.info(f"Final output: {stats_final['total_proteins']} complete protein entries")
    
    # Show failed examples
    failed_structures = LMDBHelper.get_failed_list(structure_checkpoint)
    failed_esm = LMDBHelper.get_failed_list(esm_checkpoint)
    
    if failed_structures:
        logging.info(f"\nSample failed structures:")
        for i, failed in enumerate(failed_structures[:3]):
            logging.info(f"  {i+1}. {failed['pdb']}: {failed['error']}")
    
    if failed_esm:
        logging.info(f"\nSample failed ESM extractions:")
        for i, failed in enumerate(failed_esm[:3]):
            logging.info(f"  {i+1}. {failed['pdb']}: {failed['error']}")

if __name__ == "__main__":
    main()
