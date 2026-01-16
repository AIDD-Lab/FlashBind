"""Utility functions for coordinate integration."""

from typing import Tuple, Optional, List
from rdkit import Chem
from rdkit.Chem import rdFMCS

import timeout_decorator

def extract_ids(target_id: str) -> Tuple[str, str]:
    """Extract prot_id and ligand_id from target_id.
    
    Args:
        target_id: Target ID in format {prot_id}_{ligand_id}
        
    Returns:
        Tuple of (prot_id, ligand_id)
        
    Raises:
        ValueError: If target_id format is invalid
    """
    parts = target_id.split('_')
    if len(parts) < 2:
        raise ValueError(f"Invalid target_id format: {target_id}. Expected format: prot_id_ligand_id")
    
    prot_id = parts[0]
    ligand_id = '_'.join(parts[1:])  # Handle ligand_id with underscores
    
    return prot_id, ligand_id 

def verify_mol_mapping(mol1: Chem.Mol, mol2: Chem.Mol) -> Tuple[bool, Optional[List[int]]]:
    """Verify if two molecules can be mapped using MCS.
    
    Parameters
    ----------
    mol1 : Chem.Mol
        First molecule (query)
    mol2 : Chem.Mol
        Second molecule (template)
        
    Returns
    -------
    Tuple[bool, Optional[List[int]]]
        (can_match, new_order) where new_order maps template (mol2) indices to mol1 indices
        If can_match is False, new_order is None
    """
    try:
        assert mol1.GetNumHeavyAtoms() == mol2.GetNumHeavyAtoms(), "Molecules have different number of atoms"
        # Find maximum common substructure
        mcs = rdFMCS.FindMCS([mol1, mol2], 
                            timeout=1)
        if mcs is None or not mcs.smartsString:
            mcs = rdFMCS.FindMCS([mol1, mol2], 
                            timeout=120) # Retry with longer timeout
            if mcs is None or not mcs.smartsString:
                return (False, None)
        patt = Chem.MolFromSmarts(mcs.smartsString)
        if patt is None:
            return (False, None)
        
        mol1_match = mol1.GetSubstructMatch(patt)
        mol2_match = mol2.GetSubstructMatch(patt)
        
        if not mol1_match or not mol2_match:
            return (False, None)
        
        # Check if all non-hydrogen atoms are matched
        for atom in mol1.GetAtoms():
            if atom.GetAtomicNum() != 1 and atom.GetIdx() not in mol1_match:
                return (False, None)
        
        # Build new_order: new_order[mol2_idx] = mol1_idx
        # mol1_match[i] and mol2_match[i] are corresponding atoms in mol1 and mol2
        new_order = [0] * mol2.GetNumAtoms()
        for pattern_idx in range(len(mol1_match)):
            mol1_idx = mol1_match[pattern_idx]
            mol2_idx = mol2_match[pattern_idx]
            new_order[mol2_idx] = mol1_idx
        
        return (True, new_order)
    except Exception:
        return (False, None)

def AssignBondOrdersFromTemplate(refmol, mol):
    """ assigns bond orders to a molecule based on the
        bond orders in a template molecule
    Revised from AllChem.AssignBondOrderFromTemplate(refmol, mol)
    
    Returns
    -------
    Tuple[Chem.Mol, List[int]]
        (processed molecule, new_order) where new_order maps template indices to mol indices
    """
    refmol2 = Chem.rdchem.Mol(refmol)
    mol2 = Chem.rdchem.Mol(mol)
    # do the molecules match already?
    matching = mol2.GetSubstructMatch(refmol2)
    if not matching:  # no, they don't match
        # check if bonds of mol are SINGLE
        for b in mol2.GetBonds():
            if b.GetBondType() != Chem.BondType.SINGLE:
                b.SetBondType(Chem.BondType.SINGLE)
                b.SetIsAromatic(False)
        # set the bonds of mol to SINGLE
        for b in refmol2.GetBonds():
            b.SetBondType(Chem.BondType.SINGLE)
            b.SetIsAromatic(False)
        # set atom charges to zero;
        for a in refmol2.GetAtoms():
            a.SetFormalCharge(0)
        for a in mol2.GetAtoms():
            a.SetFormalCharge(0)

        matchings = mol2.GetSubstructMatches(refmol2, uniquify=False, maxMatches=1000)
        # do the molecules match now?
        if matchings:
            matchings=matchings[:]
            for i, matching_candidate in enumerate(matchings):
                # apply matching: set bond properties
                for b in refmol.GetBonds():
                    atom1 = matching_candidate[b.GetBeginAtomIdx()]
                    atom2 = matching_candidate[b.GetEndAtomIdx()]
                    b2 = mol2.GetBondBetweenAtoms(atom1, atom2)
                    b2.SetBondType(b.GetBondType())
                    b2.SetIsAromatic(b.GetIsAromatic())
                # apply matching: set atom properties
                for a in refmol.GetAtoms():
                    a2 = mol2.GetAtomWithIdx(matching_candidate[a.GetIdx()])
                    a2.SetHybridization(a.GetHybridization())
                    a2.SetIsAromatic(a.GetIsAromatic())
                    a2.SetNumExplicitHs(a.GetNumExplicitHs())
                    a2.SetFormalCharge(a.GetFormalCharge())
                try:
                    Chem.SanitizeMol(mol2)
                    if hasattr(mol2, '__sssAtoms'):
                        mol2.__sssAtoms = None  # we don't want all bonds highlighted
                    matching = matching_candidate
                    break
                except ValueError:
                    if i == len(matchings) - 1:
                        matching = None
                    pass
        else:
            raise ValueError("No matching found")
    
    new_order = None
    if matching:
        new_order = list(matching)
    else:
        _, new_order = verify_mol_mapping(mol2, refmol)
    return mol2, new_order
