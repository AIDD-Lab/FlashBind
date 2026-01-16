import torch
import argparse
import os
from utils.inference_mol_utils import read_smiles, extract_torchdrug_feature_from_mol, generate_conformation
import pandas as pd
from tqdm import tqdm
import time
from joblib import Parallel, delayed, cpu_count

def get_mol_info(idx_and_info):
    """Process a single molecule and return the result instead of saving directly."""
    idx, (smiles, ligand_id) = idx_and_info
    try:
        mol = read_smiles(smiles)
        mol = generate_conformation(mol)
        molecule_info = extract_torchdrug_feature_from_mol(mol, has_LAS_mask=True)
        
        # Return the processed data instead of saving
        return {
            'ligand_id': ligand_id,
            'data': [mol, molecule_info],
            'success': True
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            'ligand_id': ligand_id,
            'smiles': smiles,
            'data': None,
            'success': False,
            'error': str(e)
        }

parser = argparse.ArgumentParser(description='Preprocess molecules.')
parser.add_argument("--index_csv", type=str, default="../inference_examples/test.csv",
                    help="Specify the index path for molecules.")
parser.add_argument("--save_mols_dir", type=str, default="../inference_examples/mol",
                    help="Specify where to save the processed pt.")
parser.add_argument("--num_threads", type=int, default=8,
                    help="Number of parallel jobs")
parser.add_argument("--resume", action='store_true', default=False,
                    help="Resume from existing processed molecules")
parser.add_argument("--batch_size", type=int, default=None,
                    help="Batch size for checkpoint saving (default: num_threads * 500)")
args = parser.parse_args()
os.system(f'mkdir -p {args.save_mols_dir}')

with open(args.index_csv, 'r') as f:
    content = f.readlines()
info = []
for line in content[1:]:
    smiles, ligand_id = line.strip().split(',')
    info.append([smiles, ligand_id])
info = pd.DataFrame(info, columns=['smiles', 'ligand_id'])

# Check for existing processed molecules (checkpoint resume)
output_file = os.path.join(args.save_mols_dir, 'processed_molecule.pt')
failed_file = os.path.join(args.save_mols_dir, 'failed_molecules.csv')
processed_molecules = {}
failed_molecules = []

if args.resume and os.path.exists(output_file):
    print(f"Loading existing processed molecules from {output_file}...")
    try:
        processed_molecules = torch.load(output_file)
        print(f"Found {len(processed_molecules)} already processed molecules")
    except Exception as e:
        print(f"Failed to load existing file: {e}")
        processed_molecules = {}

# Load existing failed molecules if resume is enabled
if args.resume and os.path.exists(failed_file):
    print(f"Loading existing failed molecules from {failed_file}...")
    try:
        failed_df = pd.read_csv(failed_file)
        failed_molecules = failed_df.to_dict('records')
        print(f"Found {len(failed_molecules)} previously failed molecules")
    except Exception as e:
        print(f"Failed to load existing failed file: {e}")
        failed_molecules = []

# Create a set of already processed ligand_ids (both successful and failed)
processed_ligand_ids = set(processed_molecules.keys())
failed_ligand_ids = set([f['ligand_id'] for f in failed_molecules])
all_processed_ids = processed_ligand_ids.union(failed_ligand_ids)

# Filter out already processed molecules
remaining_data = []
for i in range(len(info)):
    ligand_id = info.iloc[i].ligand_id
    if ligand_id not in all_processed_ids:
        smiles = info.iloc[i].smiles
        remaining_data.append((i, (smiles, ligand_id)))

print(f"Total molecules: {len(info)}")
print(f"Already processed successfully: {len(processed_molecules)}")
print(f"Already failed: {len(failed_molecules)}")
print(f"Remaining to process: {len(remaining_data)}")

if len(remaining_data) == 0:
    print("All molecules already processed!")
    exit(0)

if args.num_threads == 0:
    args.num_threads = int(cpu_count() * 1.5 // 2)

# Set batch size
if args.batch_size is None:
    batch_size = args.num_threads * 1000
else:
    batch_size = args.batch_size


print(f"Using {args.num_threads} parallel jobs with batch size: {batch_size}")

# Initialize counters
successful_count = len(processed_molecules)
failed_count = len(failed_molecules)

# Overall progress bar for all remaining molecules
with tqdm(total=len(remaining_data), desc="Processing molecules", unit="mol") as pbar:
    
    # Process in batches for checkpoint saving
    batch_counter = 0  # Add batch counter for save frequency control
    for batch_start in range(0, len(remaining_data), batch_size):
        batch_end = min(batch_start + batch_size, len(remaining_data))
        batch_data = remaining_data[batch_start:batch_end]
        batch_counter += 1
        
        # Process current batch using joblib parallel processing
        results = Parallel(n_jobs=args.num_threads, backend='loky')(
            delayed(get_mol_info)(data) for data in tqdm(batch_data, desc=f"Processing batch {batch_counter}")
        )
        
        # Process results and update progress
        batch_successful = 0
        batch_failed = 0
        
        for result in results:
            if result['success']:
                processed_molecules[result['ligand_id']] = result['data']
                batch_successful += 1
                successful_count += 1
            else:
                # Add failed molecule information
                failed_info = {
                    'ligand_id': result['ligand_id'],
                    'smiles': result['smiles'],
                    'error': result['error']
                }
                failed_molecules.append(failed_info)
                batch_failed += 1
                failed_count += 1
        
        # Update progress bar once per batch
        pbar.update(len(batch_data))
        pbar.set_postfix({
            'Success': successful_count,
            'Failed': failed_count,
            'Success Rate': f"{successful_count/(successful_count+failed_count)*100:.1f}%" if (successful_count+failed_count) > 0 else "0%"
        })
        
        # Save checkpoint every 10 batches instead of every batch
        if batch_counter % 10 == 0:
            torch.save(processed_molecules, output_file)
            
            # Save failed molecules to CSV
            if failed_molecules:
                failed_df = pd.DataFrame(failed_molecules)
                failed_df.to_csv(failed_file, index=False)
            
            print(f"\nCheckpoint saved at batch {batch_counter} ({successful_count} successful, {failed_count} failed)")
        
        # Optional: brief pause to avoid overwhelming the system
        time.sleep(0.1)

print(f"\nProcessing completed!")
print(f"Final results: {successful_count} successful, {failed_count} failed")

# Final save
torch.save(processed_molecules, output_file)
print(f"Saved {len(processed_molecules)} processed molecules to {output_file}")

# Save final failed molecules list
if failed_molecules:
    failed_df = pd.DataFrame(failed_molecules)
    failed_df.to_csv(failed_file, index=False)
    print(f"Saved {len(failed_molecules)} failed molecules to {failed_file}")

# Print summary information
print(f"\nSummary:")
print(f"  - Total molecules: {len(info)}")
print(f"  - Successfully processed: {successful_count}")
print(f"  - Failed: {failed_count}")
print(f"  - Success rate: {successful_count/(successful_count+failed_count)*100:.1f}%" if (successful_count+failed_count) > 0 else "N/A")
print(f"  - Output file: {output_file}")
print(f"  - Failed molecules file: {failed_file}")
print(f"  - Sample ligand_ids: {list(processed_molecules.keys())[:10]}...")  # Show first 10 only

# Show some failed examples if any
if failed_molecules:
    print(f"\nSample failed molecules:")
    for i, failed in enumerate(failed_molecules[:5]):  # Show first 5 failures
        print(f"  {i+1}. ID: {failed['ligand_id']}, SMILES: {failed['smiles'][:50]}{'...' if len(failed['smiles']) > 50 else ''}")
        print(f"     Error: {failed['error']}")
