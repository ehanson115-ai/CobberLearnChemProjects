import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# -----------------------------
# 1. Generate Noisy Dataset
# -----------------------------
np.random.seed(42)

x = np.linspace(0, 10, 50)
y_true = 2 * x + 5  # Linear "true line"

noise = np.random.normal(0, 2, size=x.shape)
y_noisy = y_true + noise

# -----------------------------
# 2. Linear Regression
# -----------------------------
X_linear = x.reshape(-1, 1)
model_linear = LinearRegression()
model_linear.fit(X_linear, y_noisy)
y_pred_linear = model_linear.predict(X_linear)

print("Linear Regression Best-Fit:")
print(f"Slope: {model_linear.coef_[0]:.4f}")
print(f"Intercept: {model_linear.intercept_:.4f}")

# -----------------------------
# 3. Quadratic Polynomial Regression
# -----------------------------
X_poly = np.column_stack((x**2, x, np.ones_like(x)))  # quadratic features
model_poly = LinearRegression()
model_poly.fit(X_poly, y_noisy)
y_pred_poly = model_poly.predict(X_poly)

print("\nPolynomial Regression Best-Fit (Quadratic):")
print(f"a (x^2 coefficient): {model_poly.coef_[0]:.4f}")
print(f"b (x coefficient): {model_poly.coef_[1]:.4f}")
print(f"c (intercept): {model_poly.intercept_:.4f}")

# -----------------------------
# 4. Plot Noisy Data + Both Fits
# -----------------------------
plt.figure(figsize=(8,5))
plt.scatter(x, y_noisy, label="Noisy Data", color="blue")
plt.plot(x, y_true, label="True Line", color="green", linewidth=2)
plt.plot(x, y_pred_linear, label="Linear Fit", color="red", linestyle='--')
plt.plot(x, y_pred_poly, label="Polynomial Fit", color="orange", linestyle='-.')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear vs Polynomial Regression Fits")
plt.legend()
plt.show()

# -----------------------------
# 5. MSE Functions
# -----------------------------
def calculate_mse_linear(m, b, x, y):
    y_guess = m * x + b
    return np.mean((y - y_guess)**2)

def calculate_mse_poly(a, b, c, x, y):
    y_guess = a * x**2 + b * x + c
    return np.mean((y - y_guess)**2)

# -----------------------------
# 6. User Input Section
# -----------------------------
print("\nTest your own linear coefficients (m, b):")
m_input = float(input("Enter slope (m): "))
b_input = float(input("Enter intercept (b): "))
mse_linear = calculate_mse_linear(m_input, b_input, x, y_noisy)
print(f"MSE for your linear guess: {mse_linear:.4f}")

print("\nTest your own quadratic coefficients (a, b, c):")
a_input = float(input("Enter a (x^2 coefficient): "))
b_input = float(input("Enter b (x coefficient): "))
c_input = float(input("Enter c (intercept): "))
mse_poly = calculate_mse_poly(a_input, b_input, c_input, x, y_noisy)
print(f"MSE for your quadratic guess: {mse_poly:.4f}")

# -----------------------------
# 7. Linear Regression Loss Landscape
# -----------------------------
m_values = np.linspace(model_linear.coef_[0]-3, model_linear.coef_[0]+3, 100)
b_values = np.linspace(model_linear.intercept_-5, model_linear.intercept_+5, 100)
M, B = np.meshgrid(m_values, b_values)
Loss_linear = np.zeros_like(M)

for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        Loss_linear[i,j] = calculate_mse_linear(M[i,j], B[i,j], x, y_noisy)

plt.figure(figsize=(8,6))
cp1 = plt.contourf(M, B, Loss_linear, 50, cmap='plasma_r')  # low MSE=yellow, high=purple
plt.colorbar(cp1, label='MSE')
plt.scatter(model_linear.coef_[0], model_linear.intercept_, color='white', marker='x', s=100, label='Best-Fit')
plt.xlabel("Slope (m)")
plt.ylabel("Intercept (b)")
plt.title("Linear Regression Loss Landscape")
plt.legend()
plt.show()

# -----------------------------
# 8. Polynomial Regression Loss Landscape (a,b) fixed c
# -----------------------------
a_values = np.linspace(model_poly.coef_[0]-0.1, model_poly.coef_[0]+0.1, 100)
b_values = np.linspace(model_poly.coef_[1]-1, model_poly.coef_[1]+1, 100)
A, B = np.meshgrid(a_values, b_values)
Loss_poly = np.zeros_like(A)
c_fixed = model_poly.intercept_

for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        Loss_poly[i,j] = calculate_mse_poly(A[i,j], B[i,j], c_fixed, x, y_noisy)

plt.figure(figsize=(8,6))
cp2 = plt.contourf(A, B, Loss_poly, 50, cmap='plasma_r')  # low MSE=yellow, high=purple
plt.colorbar(cp2, label='MSE')
plt.scatter(model_poly.coef_[0], model_poly.coef_[1], color='white', marker='x', s=100, label='Best-Fit (a,b)')
plt.xlabel("a (x^2 coefficient)")
plt.ylabel("b (x coefficient)")
plt.title(f"Polynomial Regression Loss Landscape (c fixed at {c_fixed:.2f})")
plt.legend()
plt.show()





