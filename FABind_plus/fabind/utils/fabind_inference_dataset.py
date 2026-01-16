from torch_geometric.data import Dataset
import pandas as pd
from tqdm import tqdm
import os
from pathlib import Path
from utils.resource_loader import ResourceLoader
from utils.inference_pdb_utils import extract_protein_structure, extract_esm_feature
from utils.inference_mol_utils import read_smiles, extract_torchdrug_feature_from_mol, generate_conformation
from torch_geometric.data import HeteroData
import torch


class InferenceDataset(Dataset):
    def __init__(self, index_csv, pdb_file_dir, preprocess_dir):
        super().__init__(None, None, None, None)
        # extract pair index from csv file
        with open(index_csv, 'r') as f:
            content = f.readlines()
        info = []
        for line in content[1:]:
            smiles, pdb, ligand_id = line.strip().split(',')
            info.append([smiles, pdb, ligand_id])
        info = pd.DataFrame(info, columns=['smiles', 'pdb', 'ligand_id'])

        # Initialize ResourceLoaders for lazy loading
        preprocess_path = Path(preprocess_dir)
        try:
            self.protein_loader = ResourceLoader(preprocess_path / 'processed_protein.lmdb')
        except Exception as e:
            self.protein_loader = ResourceLoader(preprocess_path / 'processed_protein.pt')
        try:
            self.molecule_loader = ResourceLoader(preprocess_path / 'processed_molecule.lmdb')
        except Exception as e:
            self.molecule_loader = ResourceLoader(preprocess_path / 'processed_molecule.pt')
   
        # Store only metadata
        self.data = []
        for i in range(len(info)):
            self.data.append({
                'smiles': info.iloc[i].smiles,
                'pdb': info.iloc[i].pdb,
                'ligand_id': info.iloc[i].ligand_id
            })

        print(f"Initialized dataset with {len(self.data)} protein-ligand pairs")


    def len(self):
        return len(self.data)

    def get(self, idx):
        # Get metadata
        metadata = self.data[idx]
        smiles = metadata['smiles']
        pdb = metadata['pdb']
        ligand_id = metadata['ligand_id']
        
        # Lazy load data using ResourceLoader
        protein_esm_feature, protein_structure = self.protein_loader.get(pdb)
        mol, molecule_info = self.molecule_loader.get(ligand_id)
        
        # Ensure all tensors are on CPU
        protein_esm_feature = protein_esm_feature.cpu()
        if isinstance(protein_structure['coords'], torch.Tensor):
            protein_structure['coords'] = protein_structure['coords'].cpu()
        
        # Extract required information
        protein_node_xyz = torch.tensor(protein_structure['coords'])[:, 1]
        protein_seq = protein_structure['seq']
        rdkit_coords, compound_node_features, input_atom_edge_list, LAS_edge_index = molecule_info
        
        # Ensure molecule tensors are on CPU
        rdkit_coords = rdkit_coords.cpu()
        compound_node_features = compound_node_features.cpu()
        input_atom_edge_list = input_atom_edge_list.cpu()
        LAS_edge_index = LAS_edge_index.cpu()
        
        n_protein_whole = protein_node_xyz.shape[0]
        n_compound = compound_node_features.shape[0]

        data = HeteroData()

        data.coord_offset = protein_node_xyz.mean(dim=0).unsqueeze(0)
        protein_node_xyz = protein_node_xyz - protein_node_xyz.mean(dim=0)
        coords_init = rdkit_coords - rdkit_coords.mean(axis=0)
        
        # compound graph
        data['compound'].node_feats = compound_node_features.float()
        data['compound', 'LAS', 'compound'].edge_index = LAS_edge_index
        data['compound'].node_coords = coords_init - coords_init.mean(dim=0)
        data['compound'].rdkit_coords = coords_init
        data['compound'].smiles = smiles
        data['compound_atom_edge_list'].x = (input_atom_edge_list[:,:2].long().contiguous() + 1).clone()
        data['LAS_edge_list'].x = (LAS_edge_index + 1).clone().t()

        data.node_xyz_whole = protein_node_xyz
        data.seq_whole = protein_seq
        data.idx = idx
        data.uid = protein_structure['name']
        data.mol = mol
        data.ligand_id = ligand_id

        # complex whole graph
        data['complex_whole_protein'].node_coords = torch.cat( # [glb_c || compound || glb_p || protein]
            (
                torch.zeros(1, 3),
                coords_init - coords_init.mean(dim=0), # for pocket prediction module, the ligand is centered at the protein center/origin
                torch.zeros(1, 3), 
                protein_node_xyz
            ), dim=0
        ).float()
        data['complex_whole_protein'].node_coords_LAS = torch.cat( # [glb_c || compound || glb_p || protein]
            (
                torch.zeros(1, 3),
                rdkit_coords,
                torch.zeros(1, 3), 
                torch.zeros_like(protein_node_xyz)
            ), dim=0
        ).float()

        segment = torch.zeros(n_protein_whole + n_compound + 2)
        segment[n_compound+1:] = 1 # compound: 0, protein: 1
        data['complex_whole_protein'].segment = segment # protein or ligand
        mask = torch.zeros(n_protein_whole + n_compound + 2)
        mask[:n_compound+2] = 1 # glb_p can be updated
        data['complex_whole_protein'].mask = mask.bool()
        is_global = torch.zeros(n_protein_whole + n_compound + 2)
        is_global[0] = 1
        is_global[n_compound+1] = 1
        data['complex_whole_protein'].is_global = is_global.bool()

        data['complex_whole_protein', 'c2c', 'complex_whole_protein'].edge_index = input_atom_edge_list[:,:2].long().t().contiguous() + 1
        data['complex_whole_protein', 'LAS', 'complex_whole_protein'].edge_index = LAS_edge_index + 1

        data['protein_whole'].node_feats = protein_esm_feature
        
        return data
    
    def close(self):
        """Close ResourceLoaders and clean up resources."""
        if hasattr(self, 'protein_loader'):
            self.protein_loader.close()
        if hasattr(self, 'molecule_loader'):
            self.molecule_loader.close()
    
    def __del__(self):
        """Destructor to automatically clean up resources."""
        self.close()

