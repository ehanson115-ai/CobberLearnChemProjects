import pubchempy as pcp

def fetch_compound_data(compound_name):
    compounds = pcp.get_compounds(compound_name, "name")

    if not compounds:
        print(f"No data found for '{compound_name}'.")
        return

    c = compounds[0]

    print("\n" + "=" * 55)
    print(f" Compound Properties: {compound_name.capitalize()}")
    print("=" * 55)

    print(f"{'Molecular Formula':<25}: {c.molecular_formula}")
    print(f"{'Molecular Weight':<25}: {c.molecular_weight}")
    print(f"{'Canonical SMILES':<25}: {c.canonical_smiles}")
    print(f"{'TPSA':<25}: {c.tpsa}")
    print(f"{'Heavy Atom Count':<25}: {c.heavy_atom_count}")

    print("=" * 55)


# Example usage
fetch_compound_data("theobromine")







