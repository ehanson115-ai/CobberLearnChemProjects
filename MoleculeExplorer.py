from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

def describe_molecule(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print("❌ Invalid SMILES string\n")
        return

    properties = {
        "Exact Molecular Weight": f"{Descriptors.ExactMolWt(mol):.4f}",
        "Hydrogen Bond Donors": rdMolDescriptors.CalcNumHBD(mol),
        "TPSA (Å²)": f"{rdMolDescriptors.CalcTPSA(mol):.2f}",
        "Number of atoms": mol.GetNumAtoms(),
        "Number of rings": rdMolDescriptors.CalcNumRings(mol),
        "Molecular formula": rdMolDescriptors.CalcMolFormula(mol),
        "Fraction of sp³ carbons (Fsp³)": f"{rdMolDescriptors.CalcFractionCSP3(mol):.2f}",
    }

    print("\nMolecule analysis")
    print("=" * 40)
    print(f"SMILES: {smiles}")
    print("-" * 40)
    for name, value in properties.items():
        print(f"{name:<35} {value}")
    print()

def main():
    print("RDKit Molecule Analyzer")
    print("Enter a SMILES string to analyze.")
    print("Type 'quit' or press Enter to exit.\n")

    while True:
        smiles = input("SMILES > ").strip()

        if smiles == "" or smiles.lower() == "quit":
            print("\nGoodbye!")
            break

        describe_molecule(smiles)

if __name__ == "__main__":
    main()




