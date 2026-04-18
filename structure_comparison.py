from Bio.PDB import PDBParser, Superimposer, PDBIO
import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
import os

# -----------------------------
# File selection
# -----------------------------
root = tk.Tk()
root.withdraw()
root.attributes('-topmost', True)

print("Select EXPERIMENTAL structure:")
exp_file = filedialog.askopenfilename(filetypes=[("PDB files", "*.pdb")])

print("Select PREDICTED structure:")
pred_file = filedialog.askopenfilename(filetypes=[("PDB files", "*.pdb")])

if not exp_file or not pred_file:
    print("File selection cancelled.")
    exit()

# -----------------------------
# Load structures
# -----------------------------
parser = PDBParser(QUIET=True)

exp_structure = parser.get_structure("experimental", exp_file)
pred_structure = parser.get_structure("predicted", pred_file)

# -----------------------------
# Basic statistics
# -----------------------------
def get_structure_stats(structure):
    return {
        "atoms": len(list(structure.get_atoms())),
        "residues": len(list(structure.get_residues())),
        "chains": len(list(structure.get_chains()))
    }

print("\n--- Experimental Structure ---")
print(get_structure_stats(exp_structure))

print("\n--- Predicted Structure ---")
print(get_structure_stats(pred_structure))

# -----------------------------
# Extract CA atoms
# -----------------------------
def get_ca_atoms(structure):
    return [res["CA"] for model in structure for chain in model for res in chain if "CA" in res]

exp_atoms = get_ca_atoms(exp_structure)
pred_atoms = get_ca_atoms(pred_structure)

min_len = min(len(exp_atoms), len(pred_atoms))
exp_atoms = exp_atoms[:min_len]
pred_atoms = pred_atoms[:min_len]

# -----------------------------
# Align + RMSD
# -----------------------------
sup = Superimposer()
sup.set_atoms(exp_atoms, pred_atoms)
sup.apply(pred_structure.get_atoms())

print(f"\nRMSD: {sup.rms:.3f} Å")

# -----------------------------
# Per-residue distances
# -----------------------------
distances = [
    np.linalg.norm(exp_atoms[i].get_coord() - pred_atoms[i].get_coord())
    for i in range(min_len)
]

# -----------------------------
# pLDDT extraction
# -----------------------------
def get_plddt_scores(structure):
    return [
        res["CA"].get_bfactor()
        for model in structure
        for chain in model
        for res in chain
        if "CA" in res
    ]

plddt_scores = get_plddt_scores(pred_structure)[:len(distances)]

# -----------------------------
# Plot: distances
# -----------------------------
plt.figure()
plt.plot(distances)
plt.xlabel("Residue Index")
plt.ylabel("Distance (Å)")
plt.title("Per-Residue Distance (Experimental vs Predicted)")
plt.show()

# -----------------------------
# Plot: pLDDT
# -----------------------------
plt.figure()
plt.plot(plddt_scores)
plt.xlabel("Residue Index")
plt.ylabel("pLDDT Score")
plt.title("AlphaFold Confidence (pLDDT)")
plt.show()

# -----------------------------
# Combined plot
# -----------------------------
plt.figure()
plt.plot(distances, label="Distance (Error)")
plt.plot(plddt_scores, label="pLDDT (Confidence)")
plt.xlabel("Residue Index")
plt.title("Error vs Confidence Comparison")
plt.legend()
plt.show()

# -----------------------------
# Correlation analysis (NEW)
# -----------------------------
correlation = np.corrcoef(plddt_scores, distances)[0, 1]
print(f"\nCorrelation (pLDDT vs Distance): {correlation:.3f}")

plt.figure()
plt.scatter(plddt_scores, distances)
plt.xlabel("pLDDT (Confidence)")
plt.ylabel("Distance (Å)")
plt.title("Correlation: Confidence vs Error")
plt.show()

# -----------------------------
# Save aligned structure
# -----------------------------
io = PDBIO()
io.set_structure(pred_structure)
io.save("aligned_predicted.pdb")

print("\nAligned structure saved as: aligned_predicted.pdb")

# -----------------------------
# PyMOL script
# -----------------------------
with open("visualize_alignment.pml", "w") as f:
    f.write(f"""
load {os.path.basename(exp_file)}, experimental
load aligned_predicted.pdb, predicted
hide everything
show cartoon
color blue, experimental
color red, predicted
align predicted, experimental
zoom
""")

print("PyMOL script saved as: visualize_alignment.pml")

# -----------------------------
# Simple HTML viewer (optional)
# -----------------------------
html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <script src="https://unpkg.com/ngl@latest/dist/ngl.js"></script>
</head>
<body>
    <div id="viewport" style="width:800px; height:600px;"></div>

    <script>
        var stage = new NGL.Stage("viewport");

        stage.loadFile("{os.path.basename(exp_file)}").then(function(o) {{
            o.addRepresentation("cartoon", {{color: "blue"}});
            stage.autoView();
        }});

        stage.loadFile("aligned_predicted.pdb").then(function(o) {{
            o.addRepresentation("cartoon", {{color: "red"}});
            stage.autoView();
        }});
    </script>
</body>
</html>
"""

with open("structure_view.html", "w") as f:
    f.write(html_content)

print("Browser visualization saved as: structure_view.html")

print("\nAll outputs generated successfully.")