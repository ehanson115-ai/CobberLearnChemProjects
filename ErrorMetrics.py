import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# Step 0: Ensure the output directory exists
output_dir = "ErrorMetrics"
os.makedirs(output_dir, exist_ok=True)

# Step 1: Given data
actual = [2, 4, 5, 4, 5, 7, 9]
predicted = [2.5, 3.5, 4, 5, 6, 8, 8]

# Step 2: Convert lists to NumPy arrays
actual_array = np.array(actual)
predicted_array = np.array(predicted)

# Step 3: Calculate residuals
residuals = predicted_array - actual_array

# Step 4: Manual calculations
mae_manual = np.mean(np.abs(residuals))
mse_manual = np.mean(residuals**2)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((actual_array - np.mean(actual_array))**2)
r2_manual = 1 - (ss_res / ss_tot)

# Step 5: scikit-learn calculations
mae_sklearn = mean_absolute_error(actual_array, predicted_array)
mse_sklearn = mean_squared_error(actual_array, predicted_array)
r2_sklearn = r2_score(actual_array, predicted_array)

# Step 6: Print results
print("Manual Calculations:")
print(f"MAE: {mae_manual}")
print(f"MSE: {mse_manual}")
print(f"R^2: {r2_manual}")

print("\nscikit-learn Calculations:")
print(f"MAE: {mae_sklearn}")
print(f"MSE: {mse_sklearn}")
print(f"R^2: {r2_sklearn}")

# Step 7: Display table of Actual | Predicted | Residuals
print("\nData Table:")
print(f"{'Actual':<10} {'Predicted':<10} {'Residual':<10}")
for a, p, r in zip(actual_array, predicted_array, residuals):
    print(f"{a:<10} {p:<10} {r:<10}")

# Step 8: Identify worst prediction (largest absolute residual)
worst_idx = np.argmax(np.abs(residuals))

# Step 9: Predicted vs Actual plot
plt.figure(figsize=(6,5))
plt.scatter(actual_array, predicted_array, marker='s', color='blue', label='Predicted')
# Highlight worst prediction in red
plt.scatter(actual_array[worst_idx], predicted_array[worst_idx], color='red', s=100, label='Worst Prediction')
plt.plot([min(actual_array), max(actual_array)],
         [min(actual_array), max(actual_array)],
         color='red', linestyle='--', label='Perfect Prediction')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs. Actual', fontsize=16)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "Predicted_vs_Actual.png"))
plt.show()

# Step 10: Residuals plot
plt.figure(figsize=(6,5))
plt.scatter(actual_array, residuals, color='pink', marker='*', s=150, label='Residuals')
# Highlight worst residual in red
plt.scatter(actual_array[worst_idx], residuals[worst_idx], color='red', s=200, label='Worst Residual')
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Actual Values', fontsize=12, fontname='Comic Sans MS')
plt.ylabel('Residuals (Predicted - Actual)', fontsize=12, fontname='Comic Sans MS')
plt.title('Residuals Plot', fontsize=16, fontname='Comic Sans MS')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "Residuals_Plot.png"))
plt.show()





