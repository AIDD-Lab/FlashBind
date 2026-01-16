import os
import torch
from torch_geometric.loader import DataLoader
from utils.logging_utils import Logger
import sys
import argparse
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from accelerate.utils import set_seed
import shlex
import time
import numpy as np
import lmdb
import pickle
import io

from tqdm import tqdm

from utils.fabind_inference_dataset_old import InferenceDataset
from utils.inference_mol_utils import write_mol
from utils.post_optim_utils import post_optimize_compound_coords
import pandas as pd
from utils.parsing import parse_train_args
from rdkit import Chem
from rdkit.Geometry import Point3D


parser = argparse.ArgumentParser(description='Train your own TankBind model.')

parser.add_argument("--ckpt", type=str, default='../checkpoints/pytorch_model.bin')
parser.add_argument("--data-path", type=str, default="/PDBbind_data/pdbbind2020",
                    help="Data path.")
parser.add_argument("--resultFolder", type=str, default="./result",
                    help="information you want to keep a record.")
parser.add_argument("--exp-name", type=str, default="",
                    help="data path.")
parser.add_argument('--seed', type=int, default=600,
                    help="seed to use.")
parser.add_argument("--batch_size", type=int, default=8,
                    help="batch size.")
parser.add_argument("--write-mol-to-file", action='store_true', default=False)
parser.add_argument('--sdf-to-mol2', action='store_true', default=False)
parser.add_argument("--infer-logging", action='store_true', default=False)
parser.add_argument("--use-clustering", action='store_true', default=False)
parser.add_argument("--dbscan-eps", type=float, default=9.0)
parser.add_argument("--dbscan-min-samples", type=int, default=2)
parser.add_argument("--choose-cluster-prob", type=float, default=0.5)
parser.add_argument("--save-rmsd-dir", type=str, default=None)
parser.add_argument("--infer-dropout", action='store_true', default=False)
parser.add_argument("--symmetric-rmsd", default=None, type=str, help="path to the raw molecule file")
parser.add_argument("--command", type=str, default=None)

parser.add_argument("--post-optim", action='store_true', default=False)
parser.add_argument('--post-optim-mode', type=int, default=0)
parser.add_argument('--post-optim-epoch', type=int, default=1000)
parser.add_argument('--sdf-output-path-post-optim', type=str, default="")

parser.add_argument('--index-csv', type=str, default=None)
parser.add_argument('--pdb-file-dir', type=str, default="")
parser.add_argument('--preprocess-dir', type=str, default="")
parser.add_argument('--instance-id', type=str, default=None, 
                    help="Unique identifier for this program instance")


infer_args = parser.parse_args()
_, train_parser = parse_train_args(test=True)

command = "fabind/main_fabind.py --batch_size 2 --label baseline --addNoise 5 --resultFolder fabind_reg --seed 224 --total-epochs 1500 --exp-name fabind_plus_regression --coord-loss-weight 1.5 --pair-distance-loss-weight 1 --pair-distance-distill-loss-weight 1 --pocket-cls-loss-weight 1 --pocket-distance-loss-weight 0.05 --pocket-radius-loss-weight 0.05 --lr 5e-5 --lr-scheduler poly_decay --n-iter 8 --mean-layers 5 --hidden-size 512 --pocket-pred-hidden-size 128 --rm-layernorm --add-attn-pair-bias --explicit-pair-embed --add-cross-attn-layer --clip-grad --expand-clength-set --cut-train-set --random-n-iter --use-ln-mlp --mlp-hidden-scale 1 --permutation-invariant --use-for-radius-pred ligand --dropout 0.1 --use-esm2-feat --dis-map-thres 15 --pocket-radius-buffer 5 --min-pocket-radius 20"
command = shlex.split(command)

args = train_parser.parse_args(command[1:])
# print(vars(infer_args))
for attr in vars(infer_args):
    # Set the corresponding attribute in args
    setattr(args, attr, getattr(infer_args, attr))
# Overwrite or set specific attributes as needed
args.tqdm_interval = 0.1
args.disable_tqdm = False

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], mixed_precision=args.mixed_precision)

# Determine instance ID for LMDB file naming
if infer_args.instance_id:
    instance_id = infer_args.instance_id
else:
    instance_id = None

# Prepare LMDB paths
lmdb_dir = args.sdf_output_path_post_optim
os.makedirs(lmdb_dir, exist_ok=True)

ligand_sdf_name = "ligand_sdf"
pocket_indices_name = "pocket_indices"
if instance_id:
    ligand_sdf_name += f"_{instance_id}"
    pocket_indices_name += f"_{instance_id}"
ligand_sdf_name += ".lmdb"
pocket_indices_name += ".lmdb"
ligand_sdf_path = os.path.join(lmdb_dir, ligand_sdf_name)
pocket_indices_path = os.path.join(lmdb_dir, pocket_indices_name)

logger = Logger(accelerator=accelerator, log_path=f'/tmp/test.log')

if instance_id:
    logger.log_message(f"Using instance ID: {instance_id}")
logger.log_message(f"LMDB paths prepared:")
logger.log_message(f"  SDF: {ligand_sdf_path}")
logger.log_message(f"  Indices: {pocket_indices_path}")

logger.log_message(f"{' '.join(sys.argv)}")

# torch.set_num_threads(16)
# # ----------without this, I could get 'RuntimeError: received 0 items of ancdata'-----------
torch.multiprocessing.set_sharing_strategy('file_system')

# Import write_mol function for SDF generation
from utils.inference_mol_utils import write_mol

def check_existing_lmdb_keys(lmdb_path):
    """
    Check existing keys in LMDB database.
    Returns set of existing keys as strings.
    """
    if not os.path.exists(lmdb_path):
        return set()
    
    try:
        env = lmdb.open(lmdb_path, readonly=True)
        existing_keys = set()
        
        with env.begin() as txn:
            cursor = txn.cursor()
            for key in cursor.iternext(keys=True, values=False):
                existing_keys.add(key.decode('utf-8'))
        
        env.close()
        return existing_keys
    except Exception as e:
        logger.log_message(f"Error reading existing LMDB keys from {lmdb_path}: {str(e)}")
        return set()

def filter_csv_by_existing_keys(csv_path, existing_keys, logger):
    """
    Filter CSV file to exclude rows that already exist in LMDB.
    Returns path to filtered CSV file or None if all data already processed.
    """
    import pandas as pd
    import tempfile
    
    # Read original CSV
    df = pd.read_csv(csv_path)
    original_count = len(df)
    logger.log_message(f"Original CSV rows: {original_count}")
    
    if not existing_keys:
        logger.log_message("No existing keys found, processing all data")
        return csv_path
    
    # Create keys from CSV data
    df['key'] = df['prot_id'].astype(str) + '_' + df['ligand_id'].astype(str)
    
    # Filter out existing keys
    filtered_df = df[~df['key'].isin(existing_keys)]
    
    # Remove the temporary key column
    filtered_df = filtered_df.drop('key', axis=1)
    
    filtered_count = len(filtered_df)
    existing_count = original_count - filtered_count
    
    logger.log_message(f"Keys already completed: {existing_count}")
    logger.log_message(f"Keys to process: {filtered_count}")
    
    if filtered_count == 0:
        logger.log_message("All data already processed, nothing to do")
        return None
    
    # Create temporary filtered CSV file
    temp_csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    filtered_df.to_csv(temp_csv.name, index=False)
    temp_csv.close()
    
    logger.log_message(f"Created filtered CSV: {temp_csv.name}")
    return temp_csv.name

def generate_sdf_content(mol, coords):
    """
    Generate SDF content as string from molecule and coordinates.
    """
    # Create a copy of the molecule to avoid modifying the original
    mol_copy = Chem.Mol(mol)
    
    # Update coordinates
    conf = mol_copy.GetConformer()
    for i in range(mol_copy.GetNumAtoms()):
        if i < len(coords):
            if hasattr(coords[i], 'item'):
                x, y, z = coords[i][0].item(), coords[i][1].item(), coords[i][2].item()
            else:
                x, y, z = float(coords[i][0]), float(coords[i][1]), float(coords[i][2])
            conf.SetAtomPosition(i, Point3D(x, y, z))
    
    # Generate SDF content as string
    sdf_content = Chem.MolToMolBlock(mol_copy)
    return sdf_content

def post_optim_mol(args, accelerator, data, com_coord_pred, com_coord_pred_per_sample_list, com_coord_offset_per_sample_list, compound_batch, keepNode_list, dataset, LAS_tmp):
    post_optim_device='cpu'
    
    # INITIALIZE REQUIRED LISTS - simplified for ligand only
    protein_residue_indices_list = []  # Store residue indices for pocket indices only
    
    for i in range(compound_batch.max().item()+1):
        i_mask = (compound_batch == i)
        com_coord_pred_i = com_coord_pred[i_mask]
        com_coord_i = data[i]['compound'].rdkit_coords

        com_coord_pred_center_i = com_coord_pred_i.mean(dim=0).reshape(1, 3)
        
        if args.post_optim:
            predict_coord, loss, rmsd = post_optimize_compound_coords(
                reference_compound_coords=com_coord_i.to(post_optim_device),
                predict_compound_coords=com_coord_pred_i.to(post_optim_device),
                LAS_edge_index=LAS_tmp[i].to(post_optim_device),
                mode=args.post_optim_mode,
                total_epoch=args.post_optim_epoch,
            )
            predict_coord = predict_coord.to(accelerator.device)
            predict_coord = predict_coord - predict_coord.mean(dim=0).reshape(1, 3) + com_coord_pred_center_i
            com_coord_pred[i_mask] = predict_coord
        
        com_coord_pred_per_sample_list.append(com_coord_pred[i_mask])
        com_coord_offset_per_sample_list.append(data[i].coord_offset)
        
        # STORE POCKET INDICES for pocket indices LMDB storage
        if i < len(keepNode_list):
            keepNode = keepNode_list[i]
            # Convert keepNode boolean mask to indices
            keepNode_indices = torch.nonzero(keepNode, as_tuple=True)[0]
            protein_residue_indices_list.append(keepNode_indices.cpu().numpy())
        else:
            protein_residue_indices_list.append(np.array([]))
        
        mol_list.append(data[i].mol)
        uid_list.append(data[i].uid)
        smiles_list.append(data[i]['compound'].smiles)
        sdf_name_list.append(f"{data[i].uid}_{data[i].ligand_id}.sdf")

    return protein_residue_indices_list


# Check existing LMDB keys and filter CSV first
logger.log_message("Checking for existing LMDB data...")
existing_keys = check_existing_lmdb_keys(ligand_sdf_path)
if not existing_keys:
    # Also check the indices database in case only one exists
    existing_keys = check_existing_lmdb_keys(pocket_indices_path)

# Filter CSV to exclude already processed items
filtered_csv_path = filter_csv_by_existing_keys(infer_args.index_csv, existing_keys, logger)

if filtered_csv_path is None:
    logger.log_message("All data already processed. Exiting.")
    exit(0)

# Create dataset with filtered CSV
dataset = InferenceDataset(filtered_csv_path, infer_args.pdb_file_dir, infer_args.preprocess_dir)
logger.log_message(f"Data points to process: {len(dataset)}")

# Clean up temporary CSV file if it was created
if filtered_csv_path != infer_args.index_csv:
    import atexit
    atexit.register(lambda: os.unlink(filtered_csv_path) if os.path.exists(filtered_csv_path) else None)

# Now create LMDB environments after filtering
logger.log_message("Creating LMDB environments...")
sdf_env = lmdb.open(ligand_sdf_path, map_size=10**12)  # 1TB max size
indices_env = lmdb.open(pocket_indices_path, map_size=10**10)  # 10GB max size
logger.log_message("LMDB environments created successfully")

num_workers = 0
data_loader = DataLoader(dataset, batch_size=args.batch_size, follow_batch=['x'], shuffle=False, pin_memory=False, num_workers=num_workers)

device = 'cuda'
from models.model import *
model = get_model(args, logger)

model = accelerator.prepare(model)

model.load_state_dict(torch.load(args.ckpt))

set_seed(args.seed)

model.eval()

logger.log_message(f"Begin inference")
start_time = time.time()

uid_list = []
smiles_list = []
sdf_name_list = []
mol_list = []
ligand_id_list = []
com_coord_pred_per_sample_list = []
com_coord_offset_per_sample_list = []

# Global lists to accumulate data across batches
protein_residue_indices_list = []

data_iter = tqdm(data_loader, mininterval=args.tqdm_interval, disable=not accelerator.is_main_process)
for batch_id, data in enumerate(data_iter):
    try:
        data = data.to(device)
        LAS_tmp = []
        for i in range(len(data)):
            LAS_tmp.append(data[i]['compound', 'LAS', 'compound'].edge_index.detach().clone())
        with torch.no_grad():
            com_coord_pred, compound_batch, pocket_coord_pred, pocket_batch, keepNode_list = model.inference(data)
                
        batch_protein_residue_indices_list = post_optim_mol(args, accelerator, data, com_coord_pred, com_coord_pred_per_sample_list, com_coord_offset_per_sample_list, compound_batch, keepNode_list, dataset, LAS_tmp=LAS_tmp)
        
        # Accumulate batch results into global lists
        protein_residue_indices_list.extend(batch_protein_residue_indices_list)
        
        # Collect ligand_id information from current batch
        for i in range(len(data)):
            ligand_id_list.append(data[i].ligand_id)
                
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(data['compound'].smiles)
        continue

if args.sdf_to_mol2:
    from utils.sdf_to_mol2 import convert_sdf_to_mol2

if args.write_mol_to_file:
    logger.log_message("Starting LMDB storage of ligand structures and pocket indices...")
    
    # Prepare data for batch storage
    sdf_data_to_store = []
    indices_data_to_store = []
    
    info = pd.DataFrame({'uid': uid_list, 'smiles': smiles_list, 'ligand_id': ligand_id_list, 'sdf_name': sdf_name_list})
    
    for i in tqdm(range(len(info)), desc="Preparing data for LMDB storage"):
        save_coords = com_coord_pred_per_sample_list[i] + com_coord_offset_per_sample_list[i]
        
        # Generate key for this complex
        key = f"{uid_list[i]}_{ligand_id_list[i]}"
        
        # Get pocket indices
        residue_indices_to_use = protein_residue_indices_list[i]
        
        # Log mode information only for the first sample
        if i == 0:
            logger.log_message(f"Storing ligand structures with {len(residue_indices_to_use)} pocket residue indices")
        
        # Generate ligand SDF content
        try:
            sdf_content = generate_sdf_content(mol_list[i], save_coords)
            
            # Prepare data for batch storage
            sdf_data_to_store.append((key, sdf_content))
            indices_data_to_store.append((key, residue_indices_to_use))
            
        except Exception as e:
            logger.log_message(f"Error generating ligand SDF for {key}: {str(e)}")
            continue
    
    # Batch store SDF data
    logger.log_message(f"Storing {len(sdf_data_to_store)} ligand SDF structures to LMDB...")
    with sdf_env.begin(write=True) as txn:
        for key, sdf_content in tqdm(sdf_data_to_store, desc="Storing SDF data"):
            txn.put(key.encode(), sdf_content.encode('utf-8'))
    
    # Batch store pocket indices
    logger.log_message(f"Storing {len(indices_data_to_store)} pocket indices to LMDB...")
    with indices_env.begin(write=True) as txn:
        for key, indices in tqdm(indices_data_to_store, desc="Storing pocket indices"):
            # Serialize numpy array to bytes
            indices_bytes = pickle.dumps(indices)
            txn.put(key.encode(), indices_bytes)
    
    logger.log_message(f"Successfully stored {len(sdf_data_to_store)} ligand structures to LMDB databases")
    logger.log_message(f"SDF database: {ligand_sdf_path}")
    logger.log_message(f"Indices database: {pocket_indices_path}")

# Close LMDB environments
sdf_env.close()
indices_env.close()

end_time = time.time()
logger.log_message(f"End infer, time spent: {end_time - start_time}")