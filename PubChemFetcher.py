# First, make sure you have PubChemPy installed:
# pip install pubchempy

import pubchempy as pcp

# Fetch the compound "theobromine" from PubChem
compound = pcp.get_compounds('theobromine', 'name')[0]

# Print molecular weight
print(f"Molecular weight: {compound.molecular_weight}")

# Print molecular formula
print(f"Molecular formula: {compound.molecular_formula}")

# Print SMILES string
print(f"SMILES: {compound.isomeric_smiles}")



