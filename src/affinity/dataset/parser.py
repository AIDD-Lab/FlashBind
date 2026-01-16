import numpy as np
import torch
from typing import Tuple, List, Optional, Union, Dict, Set
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds, rdBase
from affinity.data.repr.torchdrug import read_smiles
from affinity.utils.utils import AssignBondOrdersFromTemplate, verify_mol_mapping
import gemmi
import warnings
import timeout_decorator
warnings.filterwarnings("ignore")

# Amino acid mapping
AA_TO_INDEX = {
    'ALA': 1, 'ARG': 2, 'ASN': 3, 'ASP': 4, 'CYS': 5,
    'GLU': 6, 'GLN': 7, 'GLY': 8, 'HIS': 9, 'ILE': 10,
    'LEU': 11, 'LYS': 12, 'MET': 13, 'PHE': 14, 'PRO': 15,
    'SER': 16, 'THR': 17, 'TRP': 18, 'TYR': 19, 'VAL': 20,
    'UNK': 21  # Unknown amino acid
}

# Simplified atom type mapping for protein atoms (14 positions per residue, 1-based)
RESTYPE_TO_HEAVYATOM_NAMES = {
    'ALA': ['N', 'CA', 'C', 'O', 'CB', '',    '',    '',    '',    '',    '',    '',    '',    '', 'OXT'],
    'ARG': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'NE',  'CZ',  'NH1', 'NH2', '',    '',    '', 'OXT'],
    'ASN': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'OD1', 'ND2', '',    '',    '',    '',    '',    '', 'OXT'],
    'ASP': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'OD1', 'OD2', '',    '',    '',    '',    '',    '', 'OXT'],
    'CYS': ['N', 'CA', 'C', 'O', 'CB', 'SG',  '',    '',    '',    '',    '',    '',    '',    '', 'OXT'],
    'GLN': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'OE1', 'NE2', '',    '',    '',    '',    '', 'OXT'],
    'GLU': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'OE1', 'OE2', '',    '',    '',    '',    '', 'OXT'],
    'GLY': ['N', 'CA', 'C', 'O', '',   '',    '',    '',    '',    '',    '',    '',    '',    '', 'OXT'],
    'HIS': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'ND1', 'CD2', 'CE1', 'NE2', '',    '',    '',    '', 'OXT'],
    'ILE': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1', '',    '',    '',    '',    '',    '', 'OXT'],
    'LEU': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', '',    '',    '',    '',    '',    '', 'OXT'],
    'LYS': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'CE',  'NZ',  '',    '',    '',    '',    '', 'OXT'],
    'MET': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'SD',  'CE',  '',    '',    '',    '',    '',    '', 'OXT'],
    'PHE': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'CE1', 'CE2', 'CZ',  '',    '',    '', 'OXT'],
    'PRO': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  '',    '',    '',    '',    '',    '',    '', 'OXT'],
    'SER': ['N', 'CA', 'C', 'O', 'CB', 'OG',  '',    '',    '',    '',    '',    '',    '',    '', 'OXT'],
    'THR': ['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2', '',    '',    '',    '',    '',    '',    '', 'OXT'],
    'TRP': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2', 'OXT'],
    'TYR': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'CE1', 'CE2', 'CZ',  'OH',  '',    '', 'OXT'],
    'VAL': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', '',    '',    '',    '',    '',    '',    '', 'OXT'],
    'UNK': ['N', 'CA', 'C', 'O', '',    '',    '',    '',    '',    '',    '',    '',    '',    '', 'OXT'],
}

# Amino acid internal covalent bonds definition
# Each tuple represents a covalent bond between two atoms within the same residue
AA_COVALENT_BONDS = {
    'ALA': [('N', 'CA'), ('CA', 'C'), ('C', 'O'), ('CA', 'CB'), ('C', 'OXT')],
    'ARG': [('N', 'CA'), ('CA', 'C'), ('C', 'O'), ('CA', 'CB'), 
            ('CB', 'CG'), ('CG', 'CD'), ('CD', 'NE'), ('NE', 'CZ'),
            ('CZ', 'NH1'), ('CZ', 'NH2'), ('C', 'OXT')],
    'ASN': [('N', 'CA'), ('CA', 'C'), ('C', 'O'), ('CA', 'CB'), 
            ('CB', 'CG'), ('CG', 'OD1'), ('CG', 'ND2'), ('C', 'OXT')],
    'ASP': [('N', 'CA'), ('CA', 'C'), ('C', 'O'), ('CA', 'CB'), 
            ('CB', 'CG'), ('CG', 'OD1'), ('CG', 'OD2'), ('C', 'OXT')],
    'CYS': [('N', 'CA'), ('CA', 'C'), ('C', 'O'), ('CA', 'CB'), ('CB', 'SG'), ('C', 'OXT')],
    'GLU': [('N', 'CA'), ('CA', 'C'), ('C', 'O'), ('CA', 'CB'), 
            ('CB', 'CG'), ('CG', 'CD'), ('CD', 'OE1'), ('CD', 'OE2'), ('C', 'OXT')],
    'GLN': [('N', 'CA'), ('CA', 'C'), ('C', 'O'), ('CA', 'CB'), 
            ('CB', 'CG'), ('CG', 'CD'), ('CD', 'NE2'), ('CD', 'OE1'), ('C', 'OXT')],
    'GLY': [('N', 'CA'), ('CA', 'C'), ('C', 'O'), ('C', 'OXT')],
    'HIS': [('N', 'CA'), ('CA', 'C'), ('C', 'O'), ('CA', 'CB'), ('CB', 'CG'), ('CG', 'ND1'), 
            ('CG', 'CD2'), ('ND1', 'CE1'), ('CE1', 'NE2'), ('NE2', 'CD2'), ('C', 'OXT')],
    'ILE': [('N', 'CA'), ('CA', 'C'), ('C', 'O'), ('CA', 'CB'), 
            ('CB', 'CG1'), ('CB', 'CG2'), ('CG1', 'CD1'), ('C', 'OXT')],
    'LEU': [('N', 'CA'), ('CA', 'C'), ('C', 'O'), ('CA', 'CB'), 
            ('CB', 'CG'), ('CG', 'CD1'), ('CG', 'CD2'), ('C', 'OXT')],
    'LYS': [('N', 'CA'), ('CA', 'C'), ('C', 'O'), ('CA', 'CB'), 
            ('CB', 'CG'), ('CG', 'CD'), ('CD', 'CE'), ('CE', 'NZ'), ('C', 'OXT')],
    'MET': [('N', 'CA'), ('CA', 'C'), ('C', 'O'), ('CA', 'CB'), 
            ('CB', 'CG'), ('CG', 'SD'), ('SD', 'CE'), ('C', 'OXT')],
    'PHE': [('N', 'CA'), ('CA', 'C'), ('C', 'O'), ('CA', 'CB'), ('CB', 'CG'), ('CG', 'CD1'), 
            ('CD1', 'CE1'), ('CE1', 'CZ'), ('CZ', 'CE2'), ('CE2', 'CD2'), ('CD2', 'CG'), ('C', 'OXT')],
    'PRO': [('N', 'CA'), ('CA', 'C'), ('C', 'O'), ('CA', 'CB'), ('CB', 'CG'), 
            ('CG', 'CD'), ('CD', 'N'), ('C', 'OXT')],
    'SER': [('N', 'CA'), ('CA', 'C'), ('C', 'O'), ('CA', 'CB'), ('CB', 'OG'), ('C', 'OXT')],
    'THR': [('N', 'CA'), ('CA', 'C'), ('C', 'O'), ('CA', 'CB'), 
            ('CB', 'OG1'), ('CB', 'CG2'), ('C', 'OXT')],
    'TRP': [('N', 'CA'), ('CA', 'C'), ('C', 'O'), ('CA', 'CB'), ('CB', 'CG'), ('CG', 'CD1'), 
            ('CG', 'CD2'), ('CD1', 'NE1'), ('NE1', 'CE2'), ('CD2', 'CE2'), ('CD2', 'CE3'), 
            ('CE3', 'CZ3'), ('CZ3', 'CH2'), ('CH2', 'CZ2'), ('CZ2', 'CE2'), ('C', 'OXT')],
    'TYR': [('N', 'CA'), ('CA', 'C'), ('C', 'O'), ('CA', 'CB'), ('CB', 'CG'), ('CG', 'CD1'), 
            ('CD1', 'CE1'), ('CE1', 'CZ'), ('CZ', 'CE2'), ('CE2', 'CD2'), 
            ('CD2', 'CG'), ('CZ', 'OH'), ('C', 'OXT')],
    'VAL': [('N', 'CA'), ('CA', 'C'), ('C', 'O'), ('CA', 'CB'), 
            ('CB', 'CG1'), ('CB', 'CG2'), ('C', 'OXT')],
}

# Global precomputed bond pattern table for vectorized covalent edge construction
# Shape: [21 amino acid types, 15 atom types, 15 atom types]
BOND_PATTERN_TABLE = torch.zeros((22, 16, 16), dtype=torch.bool)

# Simplified ligand atom types
LIGAND_ATOM_TYPES = {
    'H': 1, 'B': 2, 'C': 3, 'N': 4, 'O': 5, 'F': 6, 'Mg': 7, 'Si': 8, 'P': 9, 'S': 10, 
    'Cl': 11, 'Cu': 12, 'Zn': 13, 'Se': 14, 'Br': 15, 'Sn': 16, 'I': 17, 'As': 18, 'Te': 19, 'At': 20,
    'Other': 21
}

def init_bond_pattern_table():
    """Initialize the global bond pattern table for vectorized covalent edge construction."""
    for aa_name, aa_idx in AA_TO_INDEX.items():
        bonds = AA_COVALENT_BONDS.get(aa_name, [])
        atom_names = RESTYPE_TO_HEAVYATOM_NAMES.get(aa_name, [])
        
        # Create mapping from atom name to atom type index
        atom_name_to_type = {}
        for type_idx, atom_name in enumerate(atom_names):
            if atom_name:  # Skip empty names
                atom_name_to_type[atom_name] = type_idx + 1  # +1 because atom types are 1-based
        
        # Fill bond pattern table
        for atom1_name, atom2_name in bonds:
            if atom1_name in atom_name_to_type and atom2_name in atom_name_to_type:
                type1 = atom_name_to_type[atom1_name]
                type2 = atom_name_to_type[atom2_name]
                BOND_PATTERN_TABLE[aa_idx, type1, type2] = True
                BOND_PATTERN_TABLE[aa_idx, type2, type1] = True  # Bidirectional

# Initialize the bond pattern table at module load time
init_bond_pattern_table()

def parse_protein_from_structure(structure, pocket_indices: Optional[List[int]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse protein atoms from structure object.
    
    Parameters
    ----------
    structure : gemmi.Structure
        Structure object
    pocket_indices : Optional[List[int]]
        List of residue indices to include in pocket (PDB residue_id - 1 format). 
        If None, include all residues.
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        coordinates, atom_types, residue_types, residue_indices
        All residue_indices are sequential 0-based.
    """
    # Group atoms by residue with sequential numbering
    residue_atoms = defaultdict(list)
    pocket_indices_set = None if pocket_indices is None else set(pocket_indices)
    
    # Sequential residue index counter
    sequential_residue_idx = 0
    
    for model in structure:
        for chain in model:
            for residue in chain:
                # Only process standard residues (not HETATM)
                # In gemmi, het_flag is a character: ' ' for ATOM, 'H' for HETATM
                if residue.het_flag == 'A':  # ATOM records
                    residue_name = residue.name
                    original_residue_idx = residue.seqid.num - 1  # Original PDB residue_id - 1
                    
                    # Only include specified pocket residues
                    if pocket_indices_set is None or original_residue_idx in pocket_indices_set:
                        residue_name = residue.name if residue.name in AA_TO_INDEX else 'UNK'
                        for atom in residue:
                            atom_name = atom.name
                            coords = [atom.pos.x, atom.pos.y, atom.pos.z]
                            
                            # Use sequential index instead of original index
                            residue_atoms[sequential_residue_idx].append({
                                'atom_name': atom_name,
                                'coords': coords,
                                'residue_name': residue_name,
                            })
                        
                        sequential_residue_idx += 1  # Increment sequential counter
    
    if not residue_atoms:
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    # Process each residue (same logic as before, but now using sequential indices)
    all_coords = []
    all_atom_types = []
    all_residue_types = []
    all_residue_indices = []
    
    for residue_idx in sorted(residue_atoms.keys()):  # Now sequential: 0, 1, 2, 3...
        atoms = residue_atoms[residue_idx]
        residue_name = atoms[0]['residue_name']
        
        # Get expected atom names for this residue type
        expected_atoms = RESTYPE_TO_HEAVYATOM_NAMES.get(residue_name, [])
        
        # Map actual atoms to expected positions
        atom_name_to_idx = {atom['atom_name']: i for i, atom in enumerate(atoms)}
        
        for position, expected_name in enumerate(expected_atoms):
            if expected_name and expected_name in atom_name_to_idx:
                # Found this atom
                atom_data = atoms[atom_name_to_idx[expected_name]]
                all_coords.append(atom_data['coords'])
                all_atom_types.append(position + 1)  # 1-15
                all_residue_types.append(AA_TO_INDEX[residue_name])
                all_residue_indices.append(residue_idx)  # Now sequential index
            elif expected_name:
                # Expected atom but not found - skip (could add placeholder if needed)
                pass
    
    if not all_coords:
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    return np.array(all_coords), np.array(all_atom_types), np.array(all_residue_types), np.array(all_residue_indices)

def parse_ligand_from_structure(structure: gemmi.Structure, 
                                smiles: Optional[str] = None, 
                                bond_info: Optional[Union[Set[Tuple[int, int]], Chem.Mol]] = None
                               ):
    """Parse ligand atoms from structure object.
    
    Parameters
    ----------
    structure : gemmi.Structure
        Structure object
    smiles : Optional[str]
        SMILES string for ligand structure template
    bond_info : Optional[Union[Set[Tuple[int, int]], Chem.Mol]]
        Bond information: either a set of edges (CIF) or RDKit Mol (PDB)
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, Chem.Mol]
        coordinates, atom_types, ligand_mol
    """
    ligand_atom_data = []
    
    # Extract HETATM atoms (ligand atoms)
    for model in structure:
        for chain in model:
            for residue in chain:
                # Only process HETATM records (non-protein molecules)
                if residue.het_flag == 'H':  # ligand residues
                    # Extract ALL HETATM atoms (no filtering)
                    for atom in residue:
                        if atom.element.name.strip().upper() != 'H':
                            ligand_atom_data.append(
                                (atom.serial,
                                 [atom.pos.x, atom.pos.y, atom.pos.z],
                                 atom.element.name.strip().capitalize())
                            )

    if not ligand_atom_data:
        raise ValueError("No HETATM atoms found in structure")
    
    ligand_atom_data.sort(key=lambda x: x[0])  # Sort by serial number
    _, ligand_coords, ligand_elements = zip(*ligand_atom_data)
    ligand_coords = np.array(ligand_coords)
    ligand_elements = np.array(ligand_elements)
    serial_to_index = {serial: idx for idx, (serial, _, _) in enumerate(ligand_atom_data)}
    
    # If bond_info is None (CIF case), build edges from structure.connections
    if bond_info is None:
        ligand_edges = set()
        for conn in structure.connections:
            try:
                serial1 = conn.partner1.atom.serial
                serial2 = conn.partner2.atom.serial
                if serial1 in serial_to_index and serial2 in serial_to_index:
                    idx1 = serial_to_index[serial1]
                    idx2 = serial_to_index[serial2]
                    if idx1 != idx2:
                        ligand_edges.add(tuple(sorted((idx1, idx2))))
            except Exception:
                pass
        
        if ligand_edges:
            bond_info = ligand_edges

    # Build molecule using SMILES template
    if smiles:
        try:
            ligand_mol = build_mol_with_smiles(ligand_coords, ligand_elements, smiles, bond_info)
        except Exception as e:
            raise ValueError(f"Failed to build molecule with SMILES {smiles}: {e}")
    else:
        raise ValueError("SMILES required for ligand structure reconstruction")
    
    # Extract coordinates and atom types from molecule (like SDF parsing)
    conf = ligand_mol.GetConformer(0)
    coords = np.array([[conf.GetAtomPosition(i).x,
                       conf.GetAtomPosition(i).y,
                       conf.GetAtomPosition(i).z] 
                      for i in range(ligand_mol.GetNumAtoms())])
    atom_types = np.array([LIGAND_ATOM_TYPES.get(atom.GetSymbol(), LIGAND_ATOM_TYPES['Other'])
                          for atom in ligand_mol.GetAtoms()])
    
    return coords, atom_types, ligand_mol

def standardize_element(element: str) -> str:
    if element.upper() == 'X':
        return '*'
    return element.capitalize()

def build_mol_with_smiles(coords: np.ndarray, elements: np.ndarray, smiles: str, 
                          bond_info: Optional[Union[Set[Tuple[int, int]], Chem.Mol]] = None) -> Chem.Mol:
    """Build RDKit molecule using SMILES template and coordinates with fallback strategies.
    
    Parameters
    ----------
    coords : np.ndarray
        Atomic coordinates
    elements : np.ndarray
        Element symbols
    smiles : str
        SMILES string template
    bond_info : Optional[Union[Set[Tuple[int, int]], Chem.Mol]]
        Bond information: either a set of edges (CIF) or RDKit Mol with bonds (PDB)
        
    Returns
    -------
    Chem.Mol
        RDKit molecule with coordinates and correct bond orders, 
        using the template atom order so that it matches the ligand representation
    """
    # Create template molecule from SMILES
    template_mol = read_smiles(smiles)
    
    if template_mol.GetNumAtoms() != len(elements):
        print("template mol: ", template_mol.GetNumAtoms())
        print("elements: ", len(elements))
        print("smiles: ",  smiles)
        print("elements types: ", elements)
        raise ValueError("template mol and elements have different number of atoms")

    # Create molecule from coordinates
    coords_mol = Chem.RWMol()
    conf = Chem.Conformer(len(coords))

    same_order = True
    for i, (element, coord) in enumerate(zip(elements, coords)):
        # Handle unknown atom types by mapping 'X' to wildcard atom '*'
        element_symbol = standardize_element(element)
        coords_mol.AddAtom(Chem.Atom(element_symbol))
        if element_symbol != template_mol.GetAtoms()[i].GetSymbol().capitalize():
            same_order = False
        conf.SetAtomPosition(i, coord.tolist())
    
    coords_mol.AddConformer(conf)
    assert coords_mol.GetNumAtoms() == template_mol.GetNumAtoms()

    # Strategy 1: Order matches - directly use template with coords
    if same_order:
        template_mol.AddConformer(conf)
        return template_mol
    
    # Strategy 2: Try using bond_info
    if bond_info is not None:
        if isinstance(bond_info, Chem.Mol):
            try:
                assert len(elements) == bond_info.GetNumAtoms(), "bond_info mol and elements have different number of atoms"
                for atom, element in zip(bond_info.GetAtoms(), elements):
                    assert standardize_element(atom.GetSymbol()) == standardize_element(element), \
                        f"bond_info mol {atom.GetSymbol()} and elements {element} have different atom types for atom {atom.GetIdx()}"
                # PDB case: bond_info is a Mol with correct bonds from MolFromPDBBlock
                # Verify if bond_info can map to template (check if bonds are correct)
                can_match, new_order = verify_mol_mapping(bond_info, template_mol)
                if can_match and new_order:
                    # Mapping verified, bonds and coords are correct, reorder to template
                    final_mol = Chem.RenumberAtoms(bond_info, new_order)
                    return final_mol
                # Mapping failed, try AssignBondOrders as fallback
                final_mol, new_order = AssignBondOrdersFromTemplate(template_mol, bond_info)
                if final_mol is not None and new_order:
                    final_mol = Chem.RenumberAtoms(final_mol, new_order)
                    return final_mol
            except AssertionError as e:
                print("num:", len(elements), bond_info.GetNumAtoms(), smiles)
                for i, (atom, element) in enumerate(zip(bond_info.GetAtoms(), elements)):
                    print(i, standardize_element(atom.GetSymbol()), standardize_element(element))
                    if len(elements) > bond_info.GetNumAtoms():
                        for i in range(bond_info.GetNumAtoms(), len(elements)):
                            print(i, "MISSING", standardize_element(elements[i]))
                    elif len(elements) < bond_info.GetNumAtoms():
                        for i in range(len(elements), bond_info.GetNumAtoms()):
                            print(i, standardize_element(bond_info.GetAtoms()[i].GetSymbol()), "MISSING")
                    print("what", e, len(elements), bond_info.GetNumAtoms(), smiles)
                raise e
            except Exception as e:
                pass
        
        elif isinstance(bond_info, (set, list)):
            # CIF case: bond_info is a set of edges without bond type information
            # CIF connections don't contain bond order (single/double/triple), only connectivity
            # So we skip verify and directly use AssignBondOrders
            try:
                coords_mol_with_bonds = Chem.RWMol(coords_mol.GetMol())
                for idx1, idx2 in bond_info:
                    coords_mol_with_bonds.AddBond(int(idx1), int(idx2), Chem.BondType.SINGLE)
                coords_mol_with_bonds.AddConformer(conf)
                # Try AssignBondOrders to infer correct bond types
                final_mol, new_order = AssignBondOrdersFromTemplate(template_mol, coords_mol_with_bonds.GetMol())
                if final_mol is not None:
                    final_mol = Chem.RenumberAtoms(final_mol, new_order)
                    return final_mol
            except Exception:
                pass

    # Strategy 3: Final fallback - DetermineConnectivity + AssignBondOrders
    rdDetermineBonds.DetermineConnectivity(coords_mol)
    final_mol, new_order = AssignBondOrdersFromTemplate(template_mol, coords_mol.GetMol())
    if final_mol is None or new_order is None:
        raise ValueError("Failed to assign bond orders from template")
    final_mol = Chem.RenumberAtoms(final_mol, new_order)
    return final_mol

def parse_structure_file(structure_content: str, pocket_indices: Optional[List[int]], 
                        extension: str, mode: str, smiles: Optional[str] = None):
    """Parse structure file and extract coordinates and features.
    
    Parameters
    ----------
    structure_content : str
        Structure file content
    pocket_indices : Optional[List[int]]
        List of residue indices to include in pocket. If None, include all residues.
    extension : str
        Extension of the structure file
    mode : str
        Parsing mode: 'protein' or 'complex'
    smiles : Optional[str]
        SMILES string for ligand (required for complex mode)
        
    Returns
    -------
        For protein mode: coordinates, atom_types, residue_types, residue_indices
        For complex mode: protein_coords, protein_atom_types, protein_residue_types, 
                         protein_residue_indices, ligand_coords, ligand_atom_types, ligand_mol
    """
    rdBase.DisableLog('rdApp.*')
    # Parse structure file
    try:
        if extension == 'pdb':
            structure = gemmi.read_pdb_string(structure_content)
        elif extension == 'cif':
            # For CIF files, use gemmi.cif.read_string to parse from string
            cif_block = gemmi.cif.read_string(structure_content).sole_block()
            structure = gemmi.make_structure_from_block(cif_block)
        else:
            raise ValueError(f"Unsupported structure file extension: {extension}")
    except Exception as e:
        raise ValueError(f"Failed to parse structure file: {e}")

    if mode == "protein":
        return parse_protein_from_structure(structure, pocket_indices)
    elif mode == "complex":
        # Parse protein part
        protein_coords, protein_atom_types, protein_residue_types, protein_residue_indices = \
            parse_protein_from_structure(structure, pocket_indices)
        
        # Prepare bond_info for ligand parsing
        bond_info = None
        if extension == 'pdb':
            structure_lines = structure_content.split('\n')
            ter_idx = -1
            for i, line in enumerate(structure_lines[::-1]):
                if line.startswith("TER"):
                    ter_idx = len(structure_lines) - 1 - i
                    break
            # For PDB, use MolFromPDBBlock to parse the entire PDB content
            try:
                assert ter_idx != -1, "TER record not found in PDB content"
                bond_info = Chem.MolFromPDBBlock('\n'.join(structure_lines[ter_idx+1:]), flavor=1, removeHs=True, sanitize=False)
                try:
                    bond_info = Chem.RemoveAllHs(bond_info)
                except Exception as e:
                    bond_info = Chem.RemoveAllHs(bond_info, sanitize=False)
            except Exception as e:
                pass
        # For CIF, bond_info remains None and will be handled in parse_ligand_from_structure
        
        # Parse ligand part
        ligand_coords, ligand_atom_types, ligand_mol = \
            parse_ligand_from_structure(structure, smiles, bond_info)
        
        return protein_coords, protein_atom_types, protein_residue_types, protein_residue_indices, \
               ligand_coords, ligand_atom_types, ligand_mol
    else:
        raise ValueError(f"Unsupported parsing mode: {mode}")

def parse_sdf_file(sdf_content: str) -> Tuple[np.ndarray, np.ndarray, Chem.Mol]:
    """Parse SDF content and extract ligand coordinates and atom types.
    Optionally align the SDF atom order to a reference RDKit molecule's atom order.
    
    Parameters
    ----------
    sdf_content : str
        SDF file content
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, Chem.Mol]
        coordinates, atom_types, mol_sdf
    """
    rdBase.DisableLog('rdApp.*')
    # Parse SDF to RDKit molecule and remove hydrogens for consistent indexing
    mol_sdf = Chem.MolFromMolBlock(sdf_content, sanitize=False)
    if mol_sdf is None:
        raise ValueError("RDKit failed to parse SDF content into a molecule")
    try:
        mol_sdf = Chem.RemoveAllHs(mol_sdf, sanitize=True)
    except Exception:
        mol_sdf = Chem.RemoveAllHs(mol_sdf, sanitize=False)

    # Extract coordinates and atom types from SDF molecule
    conf = mol_sdf.GetConformer(0)
    coords = np.array([[conf.GetAtomPosition(i).x,
                        conf.GetAtomPosition(i).y,
                        conf.GetAtomPosition(i).z] for i in range(mol_sdf.GetNumAtoms())])
    atom_types = np.array([
        LIGAND_ATOM_TYPES.get(atom.GetSymbol(), LIGAND_ATOM_TYPES['Other'])
        for atom in mol_sdf.GetAtoms()
    ])

    return coords, atom_types, mol_sdf