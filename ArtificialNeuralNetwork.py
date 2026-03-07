import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# -----------------------------
# Step 1: Build the Dataset
# -----------------------------
data = {
    "Compound": [
        "Methane", "Water", "Propane", "Ethanol", "Formic Acid",
        "Acetic Acid", "Butane", "Acetone", "Benzene", "Toluene", "Octane"
    ],
    "MW": [16, 18, 44, 46, 46, 60, 58, 58, 78, 92, 114],
    "BoilingPoint": [-161, 100, -42, 78, 101, 118, -1, 56, 80, 111, 125]
}

df = pd.DataFrame(data)
X = df[["MW"]]
y = df["BoilingPoint"]
X_sorted = pd.DataFrame(sorted(X["MW"]), columns=["MW"])

# -----------------------------
# Step 2: Linear Regression
# -----------------------------
lin_model = LinearRegression()
lin_model.fit(X, y)
y_pred_lin = lin_model.predict(X)
y_line = lin_model.predict(X_sorted)

plt.figure()
plt.scatter(X, y)
plt.plot(X_sorted, y_line)
plt.xlabel("Molecular Weight")
plt.ylabel("Boiling Point (°C)")
plt.title("Linear Regression: MW vs Boiling Point")
plt.savefig("linear_regression_fit.png")
plt.show()

residuals = y - y_pred_lin
plt.figure()
plt.scatter(X, residuals)
plt.axhline(0)
plt.xlabel("Molecular Weight")
plt.ylabel("Residuals")
plt.title("Linear Regression Residuals")
plt.savefig("linear_regression_residuals.png")
plt.show()

mae_lin = mean_absolute_error(y, y_pred_lin)
mse_lin = mean_squared_error(y, y_pred_lin)
r2_lin = r2_score(y, y_pred_lin)
print("Linear Regression Performance")
print("MAE:", mae_lin)
print("MSE:", mse_lin)
print("R2:", r2_lin)

# -----------------------------
# Step 3: Neural Network (Original 2-layer, 10 neurons)
# -----------------------------
original_nn = MLPRegressor(hidden_layer_sizes=(10,10), activation="relu",
                           max_iter=5000, early_stopping=False, random_state=42)
original_nn.fit(X, y)

original_preds = original_nn.predict(X_sorted)
y_pred_nn = original_nn.predict(X)
y_nn_line = original_nn.predict(X_sorted)

plt.figure()
plt.scatter(X, y)
plt.plot(X_sorted, y_nn_line)
plt.xlabel("Molecular Weight")
plt.ylabel("Boiling Point (°C)")
plt.title("Neural Network Fit")
plt.savefig("neural_network_fit.png")
plt.show()

mae_nn = mean_absolute_error(y, y_pred_nn)
mse_nn = mean_squared_error(y, y_pred_nn)
r2_nn = r2_score(y, y_pred_nn)
print("\nNeural Network Performance")
print("MAE:", mae_nn)
print("MSE:", mse_nn)
print("R2:", r2_nn)
print("Training epochs used:", original_nn.n_iter_)

# -----------------------------
# Step 4: Test with Removed Compound (Acetic Acid)
# -----------------------------
test_row = df[df["Compound"] == "Acetic Acid"]
train_df = df[df["Compound"] != "Acetic Acid"]

X_train = train_df[["MW"]]
y_train = train_df["BoilingPoint"]
X_test = test_row[["MW"]]
y_test = test_row["BoilingPoint"].values[0]

lin_model.fit(X_train, y_train)
original_nn.fit(X_train, y_train)

lin_prediction = lin_model.predict(X_test)[0]
nn_prediction = original_nn.predict(X_test)[0]

lin_error = lin_prediction - y_test
nn_error = nn_prediction - y_test

print("\nPrediction for Removed Compound (Acetic Acid)")
print("Actual BP:", y_test)
print("Linear Regression Prediction:", lin_prediction, "Error:", lin_error)
print("Neural Network Prediction:", nn_prediction, "Error:", nn_error)

plt.figure()
labels = ["Actual", "Linear Regression", "Neural Network"]
values = [y_test, lin_prediction, nn_prediction]
plt.bar(labels, values, color=["black", "blue", "red"])
plt.ylabel("Boiling Point (°C)")
plt.title("Prediction Comparison for Acetic Acid")
plt.savefig("prediction_comparison_acetic_acid.png")
plt.show()

# -----------------------------
# Step 5: Compare Linear vs Neural Network & ReLU Plot
# -----------------------------
lin_model.fit(X, y)
original_nn.fit(X, y)

lin_line = lin_model.predict(X_sorted)
nn_line = original_nn.predict(X_sorted)

plt.figure()
plt.scatter(X, y, label="Actual Data")
plt.plot(X_sorted, lin_line, label="Linear Regression")
plt.plot(X_sorted, nn_line, label="Neural Network")
plt.xlabel("Molecular Weight")
plt.ylabel("Boiling Point (°C)")
plt.title("Model Comparison: Linear vs Neural Network")
plt.legend()
plt.savefig("model_comparison.png")
plt.show()

lin_preds = lin_model.predict(X)
nn_preds = original_nn.predict(X)

print("\nFinal Model Comparison")
print("\nLinear Regression Metrics")
print("MAE:", mean_absolute_error(y, lin_preds))
print("MSE:", mean_squared_error(y, lin_preds))
print("R2:", r2_score(y, lin_preds))

print("\nNeural Network Metrics")
print("MAE:", mean_absolute_error(y, nn_preds))
print("MSE:", mean_squared_error(y, nn_preds))
print("R2:", r2_score(y, nn_preds))
print("Neural Network Training Epochs Used:", original_nn.n_iter_)

x_vals = np.linspace(-10, 10, 200)
relu = np.maximum(0, x_vals)
plt.figure()
plt.plot(x_vals, relu)
plt.xlabel("Input Value")
plt.ylabel("ReLU Output")
plt.title("ReLU Activation Function")
plt.savefig("relu_activation_function.png")
plt.show()

# -----------------------------
# Interactive Molecular Weight Prediction
# -----------------------------
print("\nInteractive Prediction: Enter a molecular weight to predict boiling point")
while True:
    try:
        mw_input = input("Enter molecular weight (or 'exit' to quit): ")
        if mw_input.lower() == "exit":
            break
        mw_input = float(mw_input)

        # Wrap in DataFrame to match training features
        mw_df = pd.DataFrame([[mw_input]], columns=["MW"])
        lin_pred = lin_model.predict(mw_df)[0]
        nn_pred = original_nn.predict(mw_df)[0]

        print(f"Linear Regression predicts: {lin_pred:.2f} °C")
        print(f"Neural Network predicts: {nn_pred:.2f} °C\n")
    except ValueError:
        print("Please enter a valid number or 'exit' to quit.")

# -----------------------------
# Step 6: Interactive Neural Network Experiments
# -----------------------------
while True:
    try:
        layers = int(input("Enter number of hidden layers (1-4, 0 to exit): "))
        if layers == 0:
            print("Exiting neural network experimentation.")
            break
        if not (1 <= layers <= 4):
            print("Please choose between 1 and 4 layers.")
            continue

        neurons = []
        for i in range(layers):
            n = int(input(f"Enter number of neurons for layer {i+1} (1-10): "))
            if not (1 <= n <= 10):
                raise ValueError("Neurons must be 1-10")
            neurons.append(n)

        nn_model = MLPRegressor(hidden_layer_sizes=tuple(neurons), activation="relu",
                                max_iter=5000, early_stopping=False, random_state=42)
        nn_model.fit(X, y)

        y_pred_sorted = nn_model.predict(X_sorted)
        y_pred_train = nn_model.predict(X)

        mae = mean_absolute_error(y, y_pred_train)
        mse = mean_squared_error(y, y_pred_train)
        r2 = r2_score(y, y_pred_train)

        print(f"\nNeural Network ({layers} layers, {neurons} neurons) Metrics:")
        print("MAE:", mae)
        print("MSE:", mse)
        print("R2:", r2)
        print("Training epochs used:", nn_model.n_iter_)

        # Plot comparison with original 2-layer network
        plt.figure()
        plt.scatter(X, y, label="Actual Data")
        plt.plot(X_sorted, original_preds, label="Original 2-layer 10-neuron NN", color="green")
        plt.plot(X_sorted, y_pred_sorted, label=f"New NN {layers} layers {neurons}", color="red")
        plt.xlabel("Molecular Weight")
        plt.ylabel("Boiling Point (°C)")
        plt.title("Neural Network Architecture Comparison")
        plt.legend()
        filename = f"nn_comparison_{layers}layers_{''.join(map(str,neurons))}neurons.png"
        plt.savefig(filename)
        plt.show()
        print(f"Comparison plot saved as: {filename}\n")

    except ValueError as ve:
        print("Invalid input:", ve)