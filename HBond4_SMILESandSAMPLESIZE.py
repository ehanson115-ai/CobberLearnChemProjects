# Updated HydrogenBonding3 code that filters invalid SMILES early and prevents the dataset from shrinking to nothing.
# It also uses a larger sample size and adds a trendline.

# ===============================
# 1. Import Libraries
# ===============================
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

import matplotlib.pyplot as plt

# ===============================
# 2. Load Dataset
# ===============================
df = pd.read_csv("curated-solubility-dataset.csv")

print("\nCOLUMNS:")
print(df.columns)

# ===============================
# 3. Identify Columns Automatically
# ===============================
SMILES_COL = None
TARGET_COL = None

for col in df.columns:
    if "smiles" in col.lower():
        SMILES_COL = col
    if "solubility" in col.lower() or "logs" in col.lower():
        TARGET_COL = col

print(f"\nUsing SMILES column: {SMILES_COL}")
print(f"Using Target column: {TARGET_COL}")

# ===============================
# 4. Clean Data
# ===============================
df = df.dropna(subset=[SMILES_COL, TARGET_COL])

print("\nOriginal dataset size:", len(df))

# Remove invalid SMILES BEFORE feature extraction
df = df[df[SMILES_COL].apply(lambda x: Chem.MolFromSmiles(x) is not None)]

print("After removing invalid SMILES:", len(df))

# Use larger sample to avoid losing too much data
df = df.sample(n=500, random_state=42)

# ===============================
# 5. Feature Extraction
# ===============================
def compute_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    num_atoms = mol.GetNumAtoms()
    aromatic_atoms = sum(atom.GetIsAromatic() for atom in mol.GetAtoms())

    return {
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
        "AromaticProportion": aromatic_atoms / num_atoms if num_atoms > 0 else 0,
        "HBD": Descriptors.NumHDonors(mol),
        "HBA": Descriptors.NumHAcceptors(mol)
    }

features = df[SMILES_COL].apply(compute_features)
features_df = pd.DataFrame(features.tolist())

# Combine features + target
data = pd.concat([features_df, df[TARGET_COL]], axis=1)

# Drop rows with missing values
data = data.dropna()

print("Final dataset size after cleaning:", len(data))

# ===============================
# 6. Define Feature Sets
# ===============================
X_no_hbond = data[[
    "MolWt",
    "LogP",
    "NumRotatableBonds",
    "AromaticProportion"
]]

X_with_hbond = data[[
    "MolWt",
    "LogP",
    "NumRotatableBonds",
    "AromaticProportion",
    "HBD",
    "HBA"
]]

y = data[TARGET_COL]

# ===============================
# 7. Train/Test Split
# ===============================
X_train1, X_test1, y_train, y_test = train_test_split(
    X_no_hbond, y, test_size=0.2, random_state=42
)

X_train2, X_test2, _, _ = train_test_split(
    X_with_hbond, y, test_size=0.2, random_state=42
)

# ===============================
# 8. Train Models
# ===============================
model1 = LinearRegression()
model2 = LinearRegression()

model1.fit(X_train1, y_train)
model2.fit(X_train2, y_train)

# ===============================
# 9. Predictions
# ===============================
y_pred1 = model1.predict(X_test1)
y_pred2 = model2.predict(X_test2)

print("\nNumber of test points:", len(y_test))

# ===============================
# 10. Evaluation
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
# 11. Coefficient Analysis
# ===============================
print("\nModel WITH Hydrogen Bonding Coefficients:")
for feature, coef in zip(X_with_hbond.columns, model2.coef_):
    print(f"{feature}: {coef:.3f}")

# ===============================
# 12. Visualization
# ===============================
plt.scatter(y_test, y_pred1, alpha=0.6, label="No H-bond")
plt.scatter(y_test, y_pred2, alpha=0.6, label="With H-bond")

plt.xlabel("Actual Solubility")
plt.ylabel("Predicted Solubility")
plt.title("Model Comparison (AqSolDB)")
plt.legend()

# Ideal line
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         linestyle='--')

plt.show()