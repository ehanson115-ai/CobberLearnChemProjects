import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from matplotlib import rcParams

# --------------------------------------------------
# Create the DataFrame
# --------------------------------------------------
data = {
    "Molecular Weight": [180, 250, 80, 300, 150, 400, 90, 200, 130, 275, 135, 220],
    "Hydrogen Bond Donors": [5, 2, 1, 1, 4, 3, 0, 2, 3, 1, 1, 3],
    "Hydrogen Bond Acceptors": [6, 3, 2, 2, 5, 4, 1, 3, 4, 2, 3, 2],
    "Water Solubility": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1]
}

molecule_labels = [f"Molecule {i}" for i in range(1, 13)]
df = pd.DataFrame(data, index=molecule_labels)

# --------------------------------------------------
# Deliberate twist: flip one solubility to see adaptation
# --------------------------------------------------
df.loc["Molecule 11", "Water Solubility"] = 1

# --------------------------------------------------
# Split features and target
# --------------------------------------------------
X = df[["Molecular Weight", "Hydrogen Bond Donors", "Hydrogen Bond Acceptors"]]
y = df["Water Solubility"]

# --------------------------------------------------
# Train the Decision Tree
# --------------------------------------------------
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

# --------------------------------------------------
# Make predictions
# --------------------------------------------------
predictions = model.predict(X)
results = pd.DataFrame({"Actual": y, "Predicted": predictions}, index=df.index)
accuracy = model.score(X, y)

print("\nPrediction Results:")
print(results)
print(f"\nTraining Accuracy: {accuracy:.2f}")

# --------------------------------------------------
# Highlight feature importance
# --------------------------------------------------
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
feature_importances = feature_importances.sort_values(ascending=False)

print("\nFeature Importances (highest → lowest):")
for feature, importance in feature_importances.items():
    print(f"{feature}: {importance:.2f}")

# --------------------------------------------------
# Visualize and save tree with red/green nodes
# --------------------------------------------------
output_dir = "DecisionTreeClassifier"
os.makedirs(output_dir, exist_ok=True)

# Override matplotlib color cycle: first color = red, second = green
rcParams['axes.prop_cycle'] = plt.cycler(color=["#d62728", "#2ca02c"])

plt.figure(figsize=(18, 11))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=["Not Water-Soluble", "Water-Soluble"],
    filled=True,
    rounded=True,
    impurity=False,  # cleaner look
    fontsize=12
)
plt.title("Decision Tree for Predicting Water Solubility", fontsize=16)
plt.tight_layout()

output_path = os.path.join(output_dir, "decision_tree_custom.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"\nCustomized decision tree image saved to: {output_path}")












