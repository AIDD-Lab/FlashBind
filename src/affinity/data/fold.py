import os
import json
import logging
import requests
import subprocess
import tempfile
import shutil
import yaml
import hashlib
import signal
import sys
from tqdm import tqdm
from Bio import PDB
from Bio.PDB import PDBIO
from Bio.SeqUtils import seq1
from affinity.data.utils import retry_request
import io

################################################################################
# Get protein structures from databases and prediction outputs of Boltz.
# 1. Search PDB for proteins and download the PDB files. (Only allow sequence edit distance <= 5 and no missing backbone atoms)
# 2. Search AlphaFold DB for proteins and download the AlphaFold structures. (Only allow sequence match perfectly)
# 3. Predict the proteins with Boltz. Collect the results and select the best model by iptm/ptm score.
# 4. Save the results to a file.
# Note: If boltz fails to predict, the protein will be predicted by AlphaFold3 server by hand.
################################################################################

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global variable to store the current subprocess
current_process = None

def signal_handler(signum, frame):
    global current_process
    logging.info("Received signal %s. Cleaning up...", signum)
    if current_process is not None:
        try:
            pgid = os.getpgid(current_process.pid)
            logging.info(f"Killing process group {pgid}...")
            os.killpg(pgid, signal.SIGTERM)
        except Exception as e:
            logging.warning(f"Failed to kill group: {e}, killing pid directly...")
            current_process.terminate()
        try:
            current_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logging.warning("Subprocess did not terminate gracefully, killing...")
            current_process.kill()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def get_gpu_count():
    """Tries to get the number of available GPUs using torch."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.device_count()
    except (ImportError, Exception):
        logging.warning("torch not found or torch.cuda.is_available() failed. Defaulting to 1 GPU.")
    return 1

class ChainSelect(PDB.Select):
    """Select specific residues from a chain for PDB extraction."""
    
    def __init__(self, chain_id, start_res, end_res):
        self.chain_id = chain_id
        self.start_res = start_res
        self.end_res = end_res
    
    def accept_chain(self, chain):
        return chain.get_id() == self.chain_id
    
    def accept_residue(self, residue):
        # Skip HETATM records (small molecules, ions, water, etc.) - only keep protein residues
        if residue.get_id()[0] != ' ':
            return False
            
        if self.start_res is None or self.end_res is None:
            # No residue range specified, accept all protein residues
            return True
        
        res_id = residue.get_id()[1]  # Get residue number
        return self.start_res <= res_id <= self.end_res

def replace_chain_id_in_pdb(pdb_content, new_chain_id='P'):
    """
    Replace all chain IDs in PDB content with the specified chain ID.
    """
    lines = pdb_content.splitlines(True)  # Keep newlines
    modified_lines = []
    
    for line in lines:
        if line.startswith("ATOM") or line.startswith("HETATM") or line.startswith("TER"):
            # Replace chain ID at position 21 (0-indexed)
            if len(line) >= 22:
                new_line = line[:21] + new_chain_id + line[22:]
                modified_lines.append(new_line)
            else:
                modified_lines.append(line)
        else:
            modified_lines.append(line)
    
    return ''.join(modified_lines)

def save_renumbered_pdb(structure, output_path, select):
    """
    Saves a structure to a PDB file by first writing to an in-memory stream
    using the standard PDBIO to preserve all record types (CONECT, etc.),
    and then post-processing this stream to renumber residues from 1 and
    fix TER records. This approach is robust and preserves all data.
    """
    # Use an in-memory StringIO object to capture the initial PDB output.
    sio = io.StringIO()
    pdbio = PDB.PDBIO()
    pdbio.set_structure(structure)
    pdbio.save(sio, select=select)
    pdb_content = sio.getvalue()
    sio.close()

    # Post-process the PDB content to fix numbering and chain ID.
    with open(output_path, 'w') as f_out:
        res_counter = 0
        last_original_res_tuple = None  # (chain_id, resseq, icode)
        last_written_res_num = 0

        for line in pdb_content.splitlines(True): # Keep newlines
            if line.startswith("ATOM") or line.startswith("HETATM"):
                chain_id = line[21]
                resseq = line[22:26]
                icode = line[26]
                
                # A residue is unique by its chain, seq number, and insertion code.
                current_original_res_tuple = (chain_id, resseq, icode)

                # If we see a new residue, increment the counter.
                if current_original_res_tuple != last_original_res_tuple:
                    res_counter += 1
                    last_original_res_tuple = current_original_res_tuple
                
                last_written_res_num = res_counter

                # Re-write the line with the new, correct residue number and chain ID "P".
                new_line = f"{line[:21]}P{res_counter:4d}{line[26:]}"
                f_out.write(new_line)

            elif line.startswith("TER"):
                # The TER card from PDBIO is mostly correct, we just fix the residue number and chain ID.
                # It should correspond to the last ATOM/HETATM of that chain.
                new_ter_line = f"{line[:21]}P{last_written_res_num:4d}{line[26:]}"
                f_out.write(new_ter_line)
                
                # Reset for the next chain. The TER card signifies a chain break.
                res_counter = 0
                last_original_res_tuple = None

            else:
                # Copy all other lines (REMARK, CONECT, END, etc.) verbatim.
                f_out.write(line)

def search_pdb_and_download(protein_name, sequence, output_path):
    """Search PDB polymer_entity, find the entity with the highest resolution that matches the sequence exactly."""
    
    # Search polymer_entity with limited results to reduce candidates
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "sequence",
                    "parameters": {
                        "identity_cutoff": 1.0,
                        "sequence_type": "protein",
                        "value": sequence
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.resolution_combined",
                        "operator": "less_or_equal",
                        "value": 3.5
                    }
                }
            ]
        },
        "return_type": "polymer_entity", 
        "request_options": {
            "sort": [{
                "sort_by": "rcsb_entry_info.resolution_combined", 
                "direction": "asc"
            }],
            "paginate": {
                "start": 0,
                "rows": 10  # Limit to first 10 results to reduce candidates
            }
        }
    }
    
    response = retry_request(requests.post, url='https://search.rcsb.org/rcsbsearch/v2/query', json=query)
    if response is None or response.status_code == 204 or not response.json().get('result_set'):
        logging.info(f"[{protein_name}] No matches found")
        return False, None
    
    candidates = response.json()['result_set']
    logging.info(f"[{protein_name}] Found {len(candidates)} candidate entities, checking for exact sequence match...")
    
    # Check each candidate for exact sequence match first, then subsequence match
    for match_priority in ["exact", "subsequence"]:
        logging.info(f"[{protein_name}] Checking for {match_priority} matches...")
        
        for candidate in candidates:
            entity_identifier = candidate['identifier']  # Format: PDB_ID_ENTITY_ID
            pdb_id = entity_identifier.split('_')[0]
            entity_id = entity_identifier.split('_')[1]
            
            try:
                # Get the complete sequence for this entity
                entity_url = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/{entity_id}"
                entity_response = retry_request(requests.get, url=entity_url)
                
                if entity_response is None or entity_response.status_code != 200:
                    logging.warning(f"[{protein_name}] Could not fetch polymer entity info for {pdb_id}/{entity_id}")
                    continue
                    
                entity_data = entity_response.json()
                
                # Extract the sequence from the entity data
                if 'entity_poly' not in entity_data or 'pdbx_seq_one_letter_code_can' not in entity_data['entity_poly']:
                    logging.warning(f"[{protein_name}] No canonical sequence found for {entity_identifier}")
                    continue
                    
                pdb_sequence = entity_data['entity_poly']['pdbx_seq_one_letter_code_can']
                
                # Check for sequence match based on current priority
                match_type = None
                start_pos = None
                end_pos = None
                
                if match_priority == "exact" and pdb_sequence == sequence:
                    match_type = "exact"
                    logging.info(f"[{protein_name}] Found exact sequence match: {entity_identifier}")
                elif match_priority == "subsequence" and pdb_sequence != sequence:
                    # Check for subsequence match
                    start_pos = pdb_sequence.find(sequence)
                    if start_pos != -1:
                        match_type = "subsequence"
                        end_pos = start_pos + len(sequence) - 1
                        logging.info(f"[{protein_name}] Found subsequence match: {entity_identifier} (positions {start_pos+1}-{end_pos+1})")
                
                if not match_type:
                    continue  # Skip this candidate for current priority
                
                # Get chain ID directly from entity data
                chain_ids_str = entity_data['entity_poly']['pdbx_strand_id']
                chain_id = chain_ids_str.split(',')[0].strip()
                
                if not chain_id:
                    logging.warning(f"[{protein_name}] No chain ID found for entity {entity_identifier}")
                    continue
                
                if match_type == "exact":
                    logging.info(f"[{protein_name}] Using chain ID: {chain_id} (full chain)")
                else:
                    logging.info(f"[{protein_name}] Using chain ID: {chain_id} (residues {start_pos+1}-{end_pos+1})")
                
                # Download PDB file
                pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
                pdb_response = retry_request(requests.get, url=pdb_url)
                
                if pdb_response is None or pdb_response.status_code != 200:
                    logging.warning(f"[{protein_name}] Could not download PDB file for {pdb_id}")
                    continue
                
                # Save PDB file temporarily
                with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as temp_pdb:
                    temp_pdb.write(pdb_response.text)
                    temp_pdb_path = temp_pdb.name
                
                try:
                    # Parse and check coordinate completeness before saving
                    parser = PDB.PDBParser(QUIET=True)
                    structure = parser.get_structure(pdb_id, temp_pdb_path)
                    
                    # Check coordinate completeness
                    if match_type == "exact":
                        is_complete, details, extracted_sequence = check_coordinate_completeness(structure, chain_id, sequence)
                    else:
                        is_complete, details, extracted_sequence = check_coordinate_completeness(structure, chain_id, sequence, start_pos+1, end_pos+1)
                    
                    if not is_complete:
                        logging.warning(f"[{protein_name}] Skipping {pdb_id} chain {chain_id}: {details}")
                        continue
                    
                    logging.info(f"[{protein_name}] {details}")
                    
                    # Save the specific chain/residue range with renumbering
                    if match_type == "exact":
                        # Save entire chain, renumbered
                        save_renumbered_pdb(structure, output_path, ChainSelect(chain_id, None, None))
                        logging.info(f"[{protein_name}] Successfully saved and renumbered {pdb_id} chain {chain_id} (full, no HOH) to {output_path}")
                    else:
                        # Save specific residue range, renumbered
                        save_renumbered_pdb(structure, output_path, ChainSelect(chain_id, start_pos + 1, end_pos + 1))
                        logging.info(f"[{protein_name}] Successfully saved and renumbered {pdb_id} chain {chain_id} residues {start_pos+1}-{end_pos+1} to {output_path}")
                    
                    return True, extracted_sequence
                    
                finally:
                    # Clean up temporary file
                    os.unlink(temp_pdb_path)
                    
            except Exception as e:
                logging.warning(f"[{protein_name}] Error processing {entity_identifier}: {e}")
                continue
    
    # If we get here, no exact match was found
    logging.info(f"[{protein_name}] No exact sequence match found among {len(candidates)} candidates")
    return False, None

def check_coordinate_completeness(structure, chain_id, expected_sequence, start_res=None, end_res=None):
    """Check if the chain has missing backbone atoms and verify sequence similarity (edit distance ≤5)."""
    try:
        chain = structure[0][chain_id]  # Get the specific chain
        backbone_atoms = ['N', 'CA', 'C', 'O']
        
        # Get all valid residues in the target range and extract sequence
        valid_residues = []
        extracted_sequence = ""
        
        for residue in chain:
            # Skip non-standard residues and water
            if residue.get_id()[0] != ' ' or residue.get_resname() == 'HOH':
                continue
            
            res_id = residue.get_id()[1]
            
            # Check if residue is in the target range
            if start_res is not None and end_res is not None:
                if not (start_res <= res_id <= end_res):
                    continue
            
            valid_residues.append(residue)
            
            # Extract amino acid sequence during traversal
            res_name = residue.get_resname()
            try:
                extracted_sequence += seq1(res_name)
            except KeyError:
                # Skip non-standard amino acids but still count the residue
                extracted_sequence += 'X'  # Use X for unknown residues
        
        if not valid_residues:
            return False, "No valid residues found", None
        
        # Check edit distance between extracted and expected sequence
        def edit_distance(s1, s2):
            """Calculate edit distance between two strings."""
            if len(s1) < len(s2):
                return edit_distance(s2, s1)
            
            if len(s2) == 0:
                return len(s1)
            
            previous_row = list(range(len(s2) + 1))
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            
            return previous_row[-1]
        
        edit_dist = edit_distance(extracted_sequence, expected_sequence)
        if edit_dist > 5:
            return False, f"Edit distance too large: {edit_dist} (extracted: {extracted_sequence}, expected: {expected_sequence})", None
        
        # Check missing backbone atoms (STRICT: any missing backbone atom fails)
        missing_atoms_count = 0
        
        for residue in valid_residues:
            # Check for missing backbone atoms
            for atom_name in backbone_atoms:
                if not residue.has_id(atom_name):
                    missing_atoms_count += 1
        
        if missing_atoms_count > 0:
            return False, f"{missing_atoms_count} backbone atoms missing", None
        
        # Good structure - return the extracted sequence
        return True, f"Good structure: edit distance {edit_dist}, no missing backbone atoms", extracted_sequence
        
    except Exception as e:
        return False, f"Error checking completeness: {e}", None

def find_uniprot_ids_by_sequence(protein_name: str, sequence: str) -> list:
    """
    Given protein sequence, return list of UniProt IDs found via MD5 checksum search.
    """
    try:
        # Calculate MD5 checksum and search UniParc
        checksum = hashlib.md5(sequence.encode('utf-8')).hexdigest().upper()
        logging.info(f"[{protein_name}] MD5 checksum: {checksum}")
        
        # Search UniParc for matching accessions
        url = "https://rest.uniprot.org/uniparc/search"
        params = {
            "fields": "accession",
            "query": f"checksum:{checksum}",
            "format": "json",
            "size": 1
        }
        
        resp = retry_request(requests.get, url=url, params=params)
        if resp is None or resp.status_code != 200:
            logging.warning(f"[{protein_name}] UniParc query failed: {resp.status_code if resp else 'No response'}")
            return []
        
        # Collect all UniProt accessions
        all_accessions = []
        results = resp.json().get("results", [])
        for result in results:
            accessions = result.get("uniProtKBAccessions", [])
            all_accessions.extend(accessions)
        
        # Remove duplicates and entries with dots, then sort by length and prefix
        unique_accessions = [acc for acc in list(set(all_accessions)) if '.' not in acc]
        len_6_p = [acc for acc in unique_accessions if len(acc) == 6 and acc.startswith('P')]
        len_6_other = [acc for acc in unique_accessions if len(acc) == 6 and not acc.startswith('P')]
        other_acc = [acc for acc in unique_accessions if len(acc) != 6]
        unique_accessions = len_6_p + len_6_other + other_acc
        # Limit to first 10 results to avoid too many candidates
        limited_accessions = unique_accessions[:10]
        logging.info(f"[{protein_name}] Found {len(unique_accessions)} UniProt IDs, using first {len(limited_accessions)}: {limited_accessions}")
        return limited_accessions
        
    except Exception as e:
        logging.error(f"[{protein_name}] Error in UniProt ID search: {e}")
        return []

def search_alphafold_and_download(protein_name: str, sequence: str, output_path: str) -> bool:
    """
    Search for exact sequence match in UniProt using MD5 checksum, then download the 
    corresponding structure from AlphaFold DB.
    """
    logging.info(f"[{protein_name}] Searching AlphaFold DB using MD5 checksum method...")
    
    try:
        # Find UniProt IDs via checksum
        uniprot_ids = find_uniprot_ids_by_sequence(protein_name, sequence)
        
        if not uniprot_ids:
            logging.info(f"[{protein_name}] No UniProt IDs found")
            return False
        
        # Verify sequences and find the best AlphaFold structure
        best_candidate = None
        best_confidence = 0.0
        
        for uniprot_id in uniprot_ids:
            try:
                # Get UniProt sequence and verify exact match
                seq_url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}"
                seq_resp = retry_request(requests.get, url=seq_url, params={"format": "json", "fields": "sequence"})
                if seq_resp is None or seq_resp.status_code != 200:
                    continue
                
                uniprot_sequence = seq_resp.json().get("sequence", {}).get("value", "")
                if uniprot_sequence != sequence:
                    logging.debug(f"[{protein_name}] Sequence mismatch for {uniprot_id}")
                    continue
                
                # Get actual confidence score and PDB content from AlphaFold
                try:
                    confidence_api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
                    conf_resp = retry_request(requests.get, url=confidence_api_url)
                    
                    if conf_resp is not None and conf_resp.status_code == 200:
                        af_url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v6.pdb"
                        pdb_resp = retry_request(requests.get, url=af_url)
                        if pdb_resp is None or pdb_resp.status_code != 200:
                            continue
                        
                        # Parse B-factors and verify exact sequence match
                        b_factors = []
                        extracted_sequence = ""
                        current_res_id = None
                        
                        for line in pdb_resp.text.split('\n'):
                            if line.startswith('ATOM') and len(line) >= 66:
                                try:
                                    # Extract residue information
                                    res_id = int(line[22:26].strip())
                                    res_name = line[17:20].strip()
                                    atom_name = line[12:16].strip()
                                    
                                    # Only process CA atoms for sequence extraction (one per residue)
                                    if atom_name == 'CA' and res_id != current_res_id:
                                        current_res_id = res_id
                                        try:
                                            extracted_sequence += seq1(res_name)
                                        except KeyError:
                                            extracted_sequence += 'X'  # Unknown amino acid
                                    
                                    # Collect B-factors from all atoms
                                    b_factor = float(line[60:66].strip())
                                    b_factors.append(b_factor)
                                    
                                except ValueError:
                                    continue
                        
                        # AlphaFold requires exact sequence match
                        if extracted_sequence != sequence:
                            logging.debug(f"[{protein_name}] AlphaFold exact sequence mismatch for {uniprot_id}: expected {sequence}, found {extracted_sequence}")
                            continue
                        
                        # Calculate average confidence from B-factors
                        confidence = sum(b_factors) / len(b_factors) / 100.0 if b_factors else 0.0
                        
                        # Update best candidate if this one is better
                        if confidence > best_confidence:
                            best_confidence = confidence
                            best_candidate = {
                                'uniprot_id': uniprot_id,
                                'confidence': confidence,
                                'pdb_content': pdb_resp.text
                            }
                            logging.info(f"[{protein_name}] New best candidate: {uniprot_id} (confidence: {confidence:.3f})")
                    else:
                        confidence = 0.0
                        
                except Exception as e:
                    logging.debug(f"[{protein_name}] Error getting confidence for {uniprot_id}: {e}")
                    continue
                
                if best_confidence == 0.0:
                    logging.debug(f"[{protein_name}] No confidence score for {uniprot_id}")
                    continue
                
            except Exception as e:
                logging.debug(f"[{protein_name}] Error verifying {uniprot_id}: {e}")
                continue
        
        # Save the best candidate if found
        if best_candidate is None:
            logging.info(f"[{protein_name}] No suitable AlphaFold structures found")
            return False
        
        # Save the best AlphaFold structure directly with chain ID "P"
        try:
            if best_candidate['confidence'] < 0.5:
                logging.info(f"[{protein_name}] Best AlphaFold structure confidence is less than 0.5, skipping.")
                return False
            # Replace chain IDs with "P" before saving
            modified_pdb_content = replace_chain_id_in_pdb(best_candidate['pdb_content'], 'P')
            with open(output_path, 'w') as f:
                f.write(modified_pdb_content)
            logging.info(f"[{protein_name}] Successfully saved best AlphaFold structure {best_candidate['uniprot_id']} (confidence: {best_candidate['confidence']:.3f}) with chain ID 'P' to {output_path}")
            return True
        except Exception as e:
            logging.warning(f"[{protein_name}] Error saving structure {best_candidate['uniprot_id']}: {e}")
            return False
        
    except Exception as e:
        logging.warning(f"[{protein_name}] Error in AlphaFold search: {e}")
        return False

def get_existing_pdb_files(output_dir):
    """
    Get a set of protein names that already have PDB files in the output directory.
    """
    existing_proteins = set()
    if os.path.exists(output_dir):
        for filename in os.listdir(output_dir):
            if filename.endswith('.pdb'):
                # Remove .pdb extension to get protein name
                protein_name = filename[:-4]
                existing_proteins.add(protein_name)
    return existing_proteins

def main():
    """
    Main function to process protein sequences.
    """
    global current_process
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--prots_json_path', type=str, default='./data/mf-pcba/prots.json')
    parser.add_argument('--output_pdb_dir', type=str, default='./data/mf-pcba/pdb')
    parser.add_argument('--work_dir', type=str, default='./data/mf-pcba/boltz_work')
    args = parser.parse_args()


    # Using user-provided paths. Note: These paths may assume the script is run from a specific directory.
    PROTS_JSON_PATH = args.prots_json_path
    OUTPUT_PDB_DIR = args.output_pdb_dir
    
    # Use fixed directories instead of temporary ones
    WORK_DIR = args.work_dir
    json_name = os.path.basename(PROTS_JSON_PATH).split('.')[0]
    yaml_input_dir = os.path.join(WORK_DIR, "yaml_", json_name)
    boltz_out_dir = os.path.join(WORK_DIR, "boltz_output", json_name)

    # Create output and work directories
    os.makedirs(OUTPUT_PDB_DIR, exist_ok=True)
    os.makedirs(yaml_input_dir, exist_ok=True)

    # Load proteins
    try:
        with open(PROTS_JSON_PATH, 'r') as f:
            proteins = json.load(f)
    except FileNotFoundError:
        logging.error(f"Could not find the protein list file at {PROTS_JSON_PATH}. Please check the path.")
        return

    logging.info(f"Loaded {len(proteins)} proteins from {PROTS_JSON_PATH}")

    # Pre-collect existing PDB files to avoid repeated os.path.exists() calls
    existing_proteins = get_existing_pdb_files(OUTPUT_PDB_DIR)
    logging.info(f"Found {len(existing_proteins)} existing PDB files, will skip these proteins")
    
    # Filter out proteins that already have PDB files
    proteins_to_process = {name: seq for name, seq in proteins.items() if name not in existing_proteins}
    logging.info(f"Will process {len(proteins_to_process)} proteins (skipped {len(proteins) - len(proteins_to_process)} existing)")

    # --- Pass 1: Search PDB and AlphaFold DB, collect proteins that need prediction ---
    proteins_to_predict = {}
    sequence_updates = {}  # Track sequence updates from PDB structures
    logging.info("--- Starting Pass 1: PDB Search ---")
    for protein_name, sequence in tqdm(proteins_to_process.items(), desc="Processing proteins"):
        logging.info(f"Processing protein: {protein_name}")
        output_pdb_path = os.path.join(OUTPUT_PDB_DIR, f"{protein_name}.pdb")
        
        # Step 1: Search PDB (allow sequence edit distance <= 5 and no missing backbone atoms)
        pdb_success, extracted_sequence = search_pdb_and_download(protein_name, sequence, output_pdb_path)
        if pdb_success:
            logging.info(f"Successfully downloaded PDB for {protein_name}.")
            # Check if sequence was updated
            if extracted_sequence and extracted_sequence != sequence:
                sequence_updates[protein_name] = extracted_sequence
                logging.info(f"[{protein_name}] Sequence updated from PDB: original length {len(sequence)}, PDB length {len(extracted_sequence)}")
            continue
        
        # Step 2: Search AlphaFold DB (exact sequence match only)
        logging.info(f"[{protein_name}] PDB search failed, trying AlphaFold DB...")
        if search_alphafold_and_download(protein_name, sequence, output_pdb_path):
            logging.info(f"Successfully downloaded AlphaFold structure for {protein_name}.")
            continue
        
        # Step 3: Queue for Boltz prediction (last resort)
        logging.info(f"[{protein_name}] PDB search failed, queued for Boltz prediction.")
        proteins_to_predict[protein_name] = sequence

    # --- Pass 2: Predict with Boltz in a single batch ---
    logging.info("--- Pass 1 Finished ---")
    if not proteins_to_predict:
        logging.info("All proteins were found in PDB or already existed. No prediction needed.")
    else:
        logging.info(f"--- Starting Pass 2: Boltz Batch Prediction for {len(proteins_to_predict)} proteins ---")

        try:
            # Create all YAML files
            for protein_name, sequence in proteins_to_predict.items():
                yaml_path = os.path.join(yaml_input_dir, f"{protein_name}.yaml")
                # This structure creates a valid YAML file for boltz, incorporating user changes.
                yaml_data = {
                    'version': 1,
                    'sequences': [{
                        'protein': {
                            'id': 'P', # Using 'P' as a generic chain ID per user request
                            'sequence': sequence
                        }
                    }]
                }
                with open(yaml_path, 'w') as f:
                    yaml.dump(yaml_data, f)
            logging.info(f"Created {len(proteins_to_predict)} YAML files in {yaml_input_dir}")

            # Construct and run one boltz command for the whole directory
            num_gpus = get_gpu_count()
            cmd = [
                'boltz', 'predict', yaml_input_dir,
                '--use_msa_server',
                '--output_format', 'pdb',
                '--out_dir', boltz_out_dir,
                '--devices', str(max(num_gpus, 1)),
                '--diffusion_samples', '1',
                '--preprocessing_threads', '4',
                '--use_potentials'
            ]
            
            logging.info(f"Running batch prediction command: {' '.join(cmd)}")
            current_process = subprocess.Popen(cmd, text=True, start_new_session=True, stdout=sys.stdout, stderr=sys.stderr, bufsize=1)
            result_code = current_process.wait()
            current_process = None

            if result_code != 0:
                logging.error(f"Boltz batch prediction failed with return code {result_code}.")
            else:
                logging.info(f"Boltz batch prediction command finished successfully.")
            
            # --- Collect results ---
            logging.info("Collecting prediction results and selecting best model by iptm/ptm score...")
            for protein_name in proteins_to_predict.keys():
                best_score = -1.0
                best_model_path = None
                best_model_index = -1
                prediction_dir = os.path.join(boltz_out_dir, f'boltz_results_{json_name}', 'predictions', protein_name)

                if not os.path.exists(prediction_dir):
                    logging.error(f"[{protein_name}] Prediction output directory not found at {prediction_dir}. Skipping.")
                    continue

                # Iterate through all 5 diffusion samples
                for i in range(1):
                    confidence_file = os.path.join(prediction_dir, f"confidence_{protein_name}_model_{i}.json")
                    model_pdb_file = os.path.join(prediction_dir, f"{protein_name}_model_{i}.pdb")

                    if os.path.exists(confidence_file) and os.path.exists(model_pdb_file):
                        try:
                            with open(confidence_file, 'r') as f:
                                confidence_data = json.load(f)
                            
                            # Per user request, check 'iptm'. For single chains, 'ptm' is the direct TM-score.
                            # We will use 'iptm' if available, otherwise fall back to 'ptm'.
                            current_score = confidence_data.get('complex_plddt', confidence_data.get('ptm', -1))

                            if current_score > best_score:
                                best_score = current_score
                                best_model_path = model_pdb_file
                                best_model_index = i

                        except (json.JSONDecodeError, KeyError) as e:
                            logging.warning(f"[{protein_name}] Could not read or parse score from {confidence_file}: {e}")
                    else:
                        # Boltz may not generate all samples if prediction fails early for some.
                        # The outputs are ordered, so if a model is missing, subsequent ones are also likely missing.
                        logging.warning(f"[{protein_name}] Missing model or confidence file for index {i}. Stopping search for this protein.")
                        break
                
                final_output_path = os.path.join(OUTPUT_PDB_DIR, f"{protein_name}.pdb")
                if best_model_path:
                    logging.info(f"[{protein_name}] Best model is model_{best_model_index} with a score (iptm/ptm) of {best_score:.4f}. Copying to {final_output_path}.")
                    # Copy the best model and replace chain ID with "P"
                    with open(best_model_path, 'r') as f_in:
                        pdb_content = f_in.read()
                    
                    # Replace chain IDs with "P"
                    modified_pdb_content = replace_chain_id_in_pdb(pdb_content, 'P')
                    
                    with open(final_output_path, 'w') as f_out:
                        f_out.write(modified_pdb_content)
                    
                    logging.info(f"[{protein_name}] Successfully saved best model with chain ID 'P' to {final_output_path}")
                else:
                    logging.error(f"[{protein_name}] Could not find any valid model to select after prediction. Please check Boltz logs.")

        except Exception as e:
            logging.error(f"An unexpected error occurred during Boltz batch prediction: {e}", exc_info=True)
        finally:
            current_process = None
            logging.info(f"Boltz prediction process completed. Working directory: {WORK_DIR}")

    # --- Update JSON file with sequence changes from PDB structures ---
    if sequence_updates:
        logging.info(f"--- Updating JSON file with {len(sequence_updates)} sequence changes from PDB structures ---")
        
        original_proteins = proteins.copy()
        # Update the proteins dictionary with new sequences
        for protein_name, new_sequence in sequence_updates.items():
            old_sequence = proteins[protein_name]
            proteins[protein_name] = new_sequence
            logging.info(f"[{protein_name}] Updated sequence: {len(old_sequence)} -> {len(new_sequence)} residues")
        
        # Write updated sequences back to JSON file
        try:
            with open(PROTS_JSON_PATH, 'w') as f:
                json.dump(proteins, f, indent=2, ensure_ascii=False)
            logging.info(f"Successfully updated {PROTS_JSON_PATH} with {len(sequence_updates)} sequence changes")
            
            # Also create a backup with timestamp
            import time
            timestamp = int(time.time())
            backup_path = f"{PROTS_JSON_PATH}.backup_{timestamp}"
            with open(backup_path, 'w') as f:
                # Write original proteins (before updates) to backup
                for protein_name, new_sequence in sequence_updates.items():
                    # Find original sequence (this is a bit tricky since we already updated it)
                    # We'll just note that this is the updated version
                    pass
                json.dump(original_proteins, f, indent=2, ensure_ascii=False)
            logging.info(f"Created backup of updated JSON at {backup_path}")
            
        except Exception as e:
            logging.error(f"Failed to update JSON file: {e}")
    else:
        logging.info("No sequence updates from PDB structures - JSON file unchanged")


if __name__ == '__main__':
    main()
