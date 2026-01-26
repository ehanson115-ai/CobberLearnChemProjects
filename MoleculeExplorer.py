# Import RDKit modules
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors


def analyze_molecule(smiles, name=None):
    """Calculate and print molecular properties for a given SMILES string."""
    molecule = Chem.MolFromSmiles(smiles)

    if not molecule:
        print(f"Error: Could not create molecule from SMILES '{smiles}'.\n")
        return

    # Descriptors
    exact_mw = Descriptors.ExactMolWt(molecule)
    h_bond_donors = rdMolDescriptors.CalcNumHBD(molecule)
    tpsa = rdMolDescriptors.CalcTPSA(molecule)
    total_atoms = molecule.GetNumAtoms()
    num_bonds = molecule.GetNumBonds()
    num_rings = rdMolDescriptors.CalcNumRings(molecule)
    rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(molecule)

    # Name for output
    if not name:
        name = smiles

    # Print results in a polished table
    print(f"\n{'=' * 50}")
    print(f"Chemical Properties of: {name}")
    print(f"{'-' * 50}")
    print(f"{'Descriptor':<25} | {'Value'}")
    print(f"{'-' * 50}")
    print(f"{'Exact Molecular Weight':<25} | {exact_mw:.3f} g/mol")
    print(f"{'Hydrogen Bond Donors':<25} | {h_bond_donors}")
    print(f"{'Topological PSA (TPSA)':<25} | {tpsa:.2f} Å²")
    print(f"{'Total Atoms':<25} | {total_atoms}")
    print(f"{'Number of Bonds':<25} | {num_bonds}")
    print(f"{'Number of Rings':<25} | {num_rings}")
    print(f"{'Rotatable Bonds':<25} | {rotatable_bonds}")
    print(f"{'=' * 50}\n")


# Main loop for multiple molecules
print("Welcome to the RDKit Molecular Property Analyzer!")
print("Enter SMILES strings of molecules to analyze (or 'quit' to exit).")

while True:
    user_input = input("\nEnter SMILES string (or 'quit'): ").strip()
    if user_input.lower() == 'quit':
        print("Exiting the analyzer. Goodbye!")
        break

    # Optional: prompt for molecule name
    molecule_name = input("Enter a name for this molecule (or press Enter to skip): ").strip()
    if molecule_name == "":
        molecule_name = None

    analyze_molecule(user_input, molecule_name)

