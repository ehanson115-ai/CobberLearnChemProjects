# RESEARCH QUESTION: How do hydrogen bonding features (donors and acceptors) influence a machine learning model’s
# ability to predict molecular solubility?

# This code reproduces the ESOL article as closely as possible (WITHOUT hydrogen bonding).

# TARGET: measured log solubility (mol/L)
# INPUTS: molecular weight, LogP, number of rotatable bonds, aromatic proportion

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
# 2. Load Dataset (ESOL)
# ===============================
url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"
df = pd.read_csv(url)

# Target variable
y = df['measured log solubility in mols per litre']


# ===============================
# 3. Compute ESOL Descriptors
# ===============================
def compute_esol_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    num_atoms = mol.GetNumAtoms()
    aromatic_atoms = sum([atom.GetIsAromatic() for atom in mol.GetAtoms()])

    return {
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
        "AromaticProportion": aromatic_atoms / num_atoms if num_atoms > 0 else 0
    }


# Apply descriptor calculation
features = df['smiles'].apply(compute_esol_descriptors)
features_df = pd.DataFrame(features.tolist())

# Combine and clean
data = pd.concat([features_df, y], axis=1).dropna()

X = data[["MolWt", "LogP", "NumRotatableBonds", "AromaticProportion"]]
y = data[y.name]

# ===============================
# 4. Train/Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# 5. Train Linear Regression Model
# ===============================
model = LinearRegression()
model.fit(X_train, y_train)

# ===============================
# 6. Predictions
# ===============================
y_pred = model.predict(X_test)

# ===============================
# 7. Evaluation
# ===============================
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("ESOL Model Performance:")
print(f"R²: {r2:.3f}")
print(f"MAE: {mae:.3f}")

# ===============================
# 8. Show Learned Equation
# ===============================
coefficients = model.coef_
intercept = model.intercept_

print("\nLearned ESOL Equation:")
for feature, coef in zip(X.columns, coefficients):
    print(f"{feature}: {coef:.3f}")
print(f"Intercept: {intercept:.3f}")

# ===============================
# 9. Visualization
# ===============================
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Solubility")
plt.ylabel("Predicted Solubility")
plt.title("ESOL Model: Predicted vs Actual")

# line of perfect prediction
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         linestyle='--')

plt.show()