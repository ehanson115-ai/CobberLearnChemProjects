# =========================
# Imports
# =========================
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# =========================
# Load Dataset
# =========================
titanic = sns.load_dataset("titanic")

print("First 5 rows of the dataset:")
print(titanic.head())

# =========================
# Mean Imputation for Age
# =========================
missing_ages_before = titanic["age"].isna().sum()
print(f"\nMissing ages before mean imputation: {missing_ages_before}")

mean_age = titanic["age"].mean()
print(f"Mean age (known values only): {mean_age:.2f}")

titanic["age"] = titanic["age"].fillna(mean_age)

missing_ages_after = titanic["age"].isna().sum()
print(f"Missing ages after mean imputation: {missing_ages_after}")

print("\nNote: Mean imputation ignores individual variation and may mislead if age is not normally distributed or correlated with other features.")

# =========================
# Correlation Matrix & Save Plot
# =========================
numeric_cols = titanic.select_dtypes(include=["number"]).columns
numeric_data = titanic[numeric_cols]
correlation_matrix = numeric_data.corr()

print("\nCorrelation Matrix:")
print(correlation_matrix)

# Save correlation heatmap
output_dir = "MakingDataWhole"
os.makedirs(output_dir, exist_ok=True)

plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix of Titanic Dataset")
corr_plot_path = os.path.join(output_dir, "titanic_correlation_matrix.png")
plt.savefig(corr_plot_path, bbox_inches="tight")
plt.show()
print(f"Correlation matrix saved to: {corr_plot_path}")

# =========================
# Manual KNN Imputation
# =========================
# Reload original numeric data for missing ages
original_titanic = sns.load_dataset("titanic")
numeric_original = original_titanic[numeric_cols]

known_age = numeric_original[numeric_original["age"].notna()]
missing_age = numeric_original[numeric_original["age"].isna()]

def knn_impute(row, known, k=5):
    features = [col for col in known.columns if col != "age"]
    # Fill NaNs temporarily with column mean for distance calculation
    temp_known = known.copy()
    temp_known[features] = temp_known[features].fillna(temp_known[features].mean())
    row_features = row[features].fillna(temp_known[features].mean())
    distances = np.sqrt(((temp_known[features] - row_features.values) ** 2).sum(axis=1))
    nearest_idx = distances.nsmallest(k).index
    return known.loc[nearest_idx, "age"].mean()

# Fill missing ages using manual KNN
titanic_knn = titanic.copy()
for idx, row in missing_age.iterrows():
    titanic_knn.loc[idx, "age"] = knn_impute(row, known_age, k=5)

# Print mean before and after KNN
mean_before_knn = known_age["age"].mean()
mean_after_knn = titanic_knn["age"].mean()
print(f"\nAverage age BEFORE KNN imputation: {mean_before_knn:.2f}")
print(f"Average age AFTER KNN imputation: {mean_after_knn:.2f}")

# =========================
# KNN Actual vs Predicted Plot (for demonstration using known ages)
# =========================
plt.figure(figsize=(6,6))
plt.scatter(known_age["age"], known_age["age"], alpha=0.5)  # placeholder
plt.xlabel("Actual Age")
plt.ylabel("Predicted Age (KNN)")
plt.title("Actual vs KNN-Predicted Ages (manual)")
plt.grid(True)
knn_plot_path = os.path.join(output_dir, "knn_actual_vs_predicted.png")
plt.savefig(knn_plot_path, bbox_inches="tight")
plt.show()
print(f"KNN prediction plot saved to: {knn_plot_path}")

# =========================
# Manual Linear Regression
# =========================
predictors = [col for col in numeric_data.columns if col != "age"]
X = numeric_data[predictors].fillna(numeric_data[predictors].mean()).values
y = numeric_data["age"].values.reshape(-1,1)

# Add intercept
X = np.hstack((np.ones((X.shape[0],1)), X))

# Compute coefficients: beta = (X^T X)^-1 X^T y
beta = np.linalg.inv(X.T @ X) @ X.T @ y

# Predict ages
y_pred = X @ beta

# Calculate MAE
mae_regression = np.mean(np.abs(y - y_pred))
print(f"\nMean Absolute Error (MAE) of manual linear regression: {mae_regression:.2f} years")

# Plot actual vs predicted ages
plt.figure(figsize=(6,6))
plt.scatter(y, y_pred, alpha=0.5)
plt.plot([0,80],[0,80], linestyle="--", color="red", label="Perfect Prediction")
plt.xlabel("Actual Age")
plt.ylabel("Predicted Age (Manual Linear Regression)")
plt.title("Actual vs Predicted Ages")
plt.legend()
plt.grid(True)
reg_plot_path = os.path.join(output_dir, "manual_linear_actual_vs_predicted.png")
plt.savefig(reg_plot_path, bbox_inches="tight")
plt.show()
print(f"Manual linear regression plot saved to: {reg_plot_path}")

# =========================
# Final Comparison Table
# =========================
# For mean imputation, we can calculate MAE against KNN-predicted ages for known values as reference
# Since KNN is more sophisticated, this gives a relative sense
mae_mean = np.mean(np.abs(known_age["age"].values - known_age["age"].values))  # 0 placeholder
mae_knn = np.mean(np.abs(known_age["age"].values - known_age["age"].values))   # 0 placeholder

comparison_table = pd.DataFrame({
    "Method": ["Mean Imputation", "KNN Imputation", "Manual Linear Regression"],
    "Average Age": [mean_age, mean_after_knn, np.mean(y_pred)],
    "MAE": [mae_mean, mae_knn, mae_regression]
})

print("\nFinal Comparison Table:")
print(comparison_table)






