# This code reproduces the approach of the ESOL model while extending it by explicitly incorporating hydrogen
# bonding descriptors (hydrogen bond donors and acceptors). By comparing model performance with and without these
# features, the study evaluates the contribution of hydrogen bonding to solubility prediction.

# TARGET: measured log solubility (mol/L)
# INPUTS: molecular weight, LogP, number of rotatable bonds, aromatic proportion, hydrogen bond acceptors, hydrogen
# bond donors

# The R squared value increased from 0.744 to 0.754 when predicting WITH hydrogen bonding. MAE also decreased from
# 0.822 to 0.806, indicating that hydrogen bonding descriptors improves solubility prediction.

# ===============================
# 1. Import Libraries
# ===============================
import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import Descriptors

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

import matplotlib.pyplot as plt

# ===============================
# 2. Load ESOL Dataset
# ===============================
url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"
df = pd.read_csv(url)

print("Dataset preview:")
print(df.head())

# Target variable (solubility)
y = df['measured log solubility in mols per litre']


# ===============================
# 3. Compute Molecular Features
# ===============================
def compute_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    return {
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
        "AromaticProportion": Descriptors.NumAromaticRings(mol) / max(1, mol.GetNumAtoms()),
        "HBD": Descriptors.NumHDonors(mol),  # hydrogen bond donors
        "HBA": Descriptors.NumHAcceptors(mol)  # hydrogen bond acceptors
    }


# Apply feature extraction
features = df['smiles'].apply(compute_features)
features_df = pd.DataFrame(features.tolist())

# Remove missing rows
data = pd.concat([features_df, y], axis=1).dropna()

# ===============================
# 4. Define Feature Sets
# ===============================
# WITHOUT hydrogen bonding
X_no_hbond = data[["MolWt", "LogP", "NumRotatableBonds", "AromaticProportion"]]

# WITH hydrogen bonding
X_with_hbond = data[["MolWt", "LogP", "NumRotatableBonds", "AromaticProportion", "HBD", "HBA"]]

y = data[y.name]

# ===============================
# 5. Train/Test Split
# ===============================
X_train1, X_test1, y_train, y_test = train_test_split(X_no_hbond, y, test_size=0.2, random_state=42)
X_train2, X_test2, _, _ = train_test_split(X_with_hbond, y, test_size=0.2, random_state=42)

# ===============================
# 6. Train Models
# ===============================
model1 = LinearRegression()
model2 = LinearRegression()

model1.fit(X_train1, y_train)
model2.fit(X_train2, y_train)

# ===============================
# 7. Predictions
# ===============================
y_pred1 = model1.predict(X_test1)
y_pred2 = model2.predict(X_test2)


# ===============================
# 8. Evaluate Models
# ===============================
def evaluate(y_true, y_pred, name):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    print(f"\n{name}")
    print(f"R²: {r2:.3f}")
    print(f"MAE: {mae:.3f}")


evaluate(y_test, y_pred1, "WITHOUT Hydrogen Bonding")
evaluate(y_test, y_pred2, "WITH Hydrogen Bonding")

# ===============================
# 9. Visualization
# ===============================
plt.scatter(y_test, y_pred1, label="No H-bond", alpha=0.6)
plt.scatter(y_test, y_pred2, label="With H-bond", alpha=0.6)

plt.xlabel("Actual Solubility")
plt.ylabel("Predicted Solubility")
plt.title("Model Comparison")
plt.legend()

plt.show()