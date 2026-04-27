# Keeps scatter plot for linear regression model, but converts the Random Forest output into categories (bins) and
# builds a confusion matrix from those categories.

# ===============================
# 1. Import Libraries
# ===============================
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

import matplotlib.pyplot as plt

# ===============================
# 2. Load Dataset
# ===============================
df = pd.read_csv("curated-solubility-dataset.csv")

# ===============================
# 3. Identify Columns
# ===============================
SMILES_COL = None
TARGET_COL = None

for col in df.columns:
    if "smiles" in col.lower():
        SMILES_COL = col
    if "solubility" in col.lower() or "logs" in col.lower():
        TARGET_COL = col

print("\n==============================")
print(" DATASET INFO")
print("==============================")
print(f"SMILES column: {SMILES_COL}")
print(f"Target column: {TARGET_COL}")

# ===============================
# 4. Clean Data
# ===============================
df = df.dropna(subset=[SMILES_COL, TARGET_COL])
df = df[df[SMILES_COL].apply(lambda x: Chem.MolFromSmiles(x) is not None)]
df = df.sample(n=1000, random_state=42)

print(f"Dataset size after cleaning: {len(df)}")

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

data = pd.concat([features_df, df[TARGET_COL]], axis=1).dropna()

print(f"Final usable dataset: {len(data)}")

# ===============================
# 6. Feature Sets
# ===============================
X_no_hbond = data[["MolWt", "LogP", "NumRotatableBonds", "AromaticProportion"]]

X_with_hbond = data[[
    "MolWt", "LogP", "NumRotatableBonds",
    "AromaticProportion", "HBD", "HBA"
]]

X_no_logp = data[[
    "MolWt", "NumRotatableBonds",
    "AromaticProportion", "HBD", "HBA"
]]

y = data[TARGET_COL]

# ===============================
# 7. Train/Test Splits
# ===============================
X_train1, X_test1, y_train, y_test = train_test_split(
    X_no_hbond, y, test_size=0.2, random_state=42
)

X_train2, X_test2, _, _ = train_test_split(
    X_with_hbond, y, test_size=0.2, random_state=42
)

X_train3, X_test3, y_train3, y_test3 = train_test_split(
    X_no_logp, y, test_size=0.2, random_state=42
)

print(f"Test set size: {len(y_test)}")

# ===============================
# 8. Train Models
# ===============================
model1 = LinearRegression()
model2 = LinearRegression()
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model_no_logp = RandomForestRegressor(n_estimators=100, random_state=42)

model1.fit(X_train1, y_train)
model2.fit(X_train2, y_train)
rf_model.fit(X_train2, y_train)
rf_model_no_logp.fit(X_train3, y_train3)

# ===============================
# 9. Predictions
# ===============================
y_pred1 = model1.predict(X_test1)
y_pred2 = model2.predict(X_test2)
y_pred_rf = rf_model.predict(X_test2)
y_pred_rf_no_logp = rf_model_no_logp.predict(X_test3)

# ===============================
# 10. Evaluation Table
# ===============================
def get_metrics(y_true, y_pred):
    return round(r2_score(y_true, y_pred), 3), round(mean_absolute_error(y_true, y_pred), 3)

results = pd.DataFrame({
    "Model": [
        "Linear (No H-bond)",
        "Linear (With H-bond)",
        "Random Forest (With H-bond)",
        "Random Forest (No LogP)"
    ],
    "R²": [
        get_metrics(y_test, y_pred1)[0],
        get_metrics(y_test, y_pred2)[0],
        get_metrics(y_test, y_pred_rf)[0],
        get_metrics(y_test3, y_pred_rf_no_logp)[0]
    ],
    "MAE": [
        get_metrics(y_test, y_pred1)[1],
        get_metrics(y_test, y_pred2)[1],
        get_metrics(y_test, y_pred_rf)[1],
        get_metrics(y_test3, y_pred_rf_no_logp)[1]
    ]
})

print("\n==============================")
print(" MODEL PERFORMANCE")
print("==============================")
print(results.to_string(index=False))

# ===============================
# 11. Linear Coefficients
# ===============================
coef_df = pd.DataFrame({
    "Feature": X_with_hbond.columns,
    "Coefficient": model2.coef_
}).sort_values(by="Coefficient", key=abs, ascending=False)

print("\n==============================")
print(" LINEAR MODEL COEFFICIENTS")
print("==============================")
print(coef_df.to_string(index=False))

# ===============================
# 12. Random Forest Importance
# ===============================
importance_df = pd.DataFrame({
    "Feature": X_with_hbond.columns,
    "Importance": rf_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\n==============================")
print(" RANDOM FOREST FEATURE IMPORTANCE")
print("==============================")
print(importance_df.to_string(index=False))

# ===============================
# 13. Visualization
# ===============================

# ---- Scatter Plot (Linear Models Only) ----
plt.figure(figsize=(7,6))

plt.scatter(y_test, y_pred1, alpha=0.5, label="Linear No H-bond")
plt.scatter(y_test, y_pred2, alpha=0.5, label="Linear With H-bond")

plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         linestyle='--')

plt.xlabel("Actual Solubility")
plt.ylabel("Predicted Solubility")
plt.title("Linear Regression: Actual vs Predicted")
plt.legend()

plt.show()


# ---- Confusion Matrix for Random Forest ----
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Convert continuous values into categories (bins)
# You can adjust number of bins if needed
num_bins = 3

y_test_binned = pd.qcut(y_test, q=num_bins, labels=False)
y_pred_rf_binned = pd.qcut(y_pred_rf, q=num_bins, labels=False)

# Compute confusion matrix
cm = confusion_matrix(y_test_binned, y_pred_rf_binned)

# Display
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

plt.title("Random Forest Confusion Matrix (Binned Solubility)")
plt.show()