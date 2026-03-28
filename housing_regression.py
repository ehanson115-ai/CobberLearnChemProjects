# =========================================
# CALIFORNIA HOUSING PRICE PREDICTOR
# =========================================

# =========================================
# 1. Import Libraries
# =========================================
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# =========================================
# 2. Load Dataset
# =========================================
# Load dataset and convert to DataFrame
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df["MedHouseValue"] = housing.target

# =========================================
# 3. Visualizations
# =========================================
print("\nGenerating visualizations...")

# Histograms
df.hist(figsize=(12, 10))
plt.tight_layout()
plt.savefig("feature_histograms.png")
plt.close()

# Correlation heatmap
corr_matrix = df.corr()

plt.figure(figsize=(10, 8))
plt.imshow(corr_matrix)
plt.colorbar()
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
plt.title("Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.close()

# Print most important correlations
print("\nTop correlations with house value:")
print(corr_matrix["MedHouseValue"].sort_values(ascending=False))

# =========================================
# 4. Prepare Data
# =========================================
# Separate features (X) and target (y)
X = df.drop("MedHouseValue", axis=1)
y = df["MedHouseValue"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale data for SVR model
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================================
# 5. Train Models
# =========================================
print("\nTraining models...")

lin_reg = LinearRegression()
tree = DecisionTreeRegressor(random_state=42)
forest = RandomForestRegressor(random_state=42)
svr = SVR()

# Fit models
lin_reg.fit(X_train, y_train)
tree.fit(X_train, y_train)
forest.fit(X_train, y_train)
svr.fit(X_train_scaled, y_train)

# =========================================
# 6. Evaluate Models
# =========================================
print("\n===== MODEL PERFORMANCE (R²) =====")

scores = {
    "Linear Regression": r2_score(y_test, lin_reg.predict(X_test)),
    "Decision Tree": r2_score(y_test, tree.predict(X_test)),
    "Random Forest": r2_score(y_test, forest.predict(X_test)),
    "SVR": r2_score(y_test, svr.predict(X_test_scaled))
}

for model, score in scores.items():
    print(f"{model}: {score:.3f}")

# Identify best model
best_model_name = max(scores, key=scores.get)
print(f"\nBest model: {best_model_name}")


# =========================================
# R² Scores for each model
# =========================================
# Linear Regression: 0.575787706032451
# Decision Tree: 0.622075845135081
# Random Forest: 0.8051230593157366
# SVR: 0.7275628923016776

# The Random Forest model performed the best because it combines multiple decision trees. allowing it to capture
# complex, nonlinear relationships in the data while reducing overfitting.

# =========================================
# 7. Predicted vs Actual Plot
# =========================================
print("\nCreating prediction plot...")

y_pred_best = forest.predict(X_test)

plt.figure()
plt.scatter(y_test, y_pred_best)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Predicted vs Actual Prices")
plt.savefig("predicted_vs_actual.png")
plt.close()

# =========================================
# 8. Example Predictions
# =========================================
print("\n===== SAMPLE PREDICTIONS =====")

sample_houses = pd.DataFrame([
    [8.0, 20, 6.0, 1.0, 1000, 3.0, 37.5, -122.0],
    [4.0, 30, 5.0, 1.2, 1500, 2.5, 36.0, -119.0],
    [2.0, 40, 4.0, 1.5, 2000, 2.0, 34.0, -118.0]
], columns=X.columns)

preds = forest.predict(sample_houses)

for i, price in enumerate(preds):
    print(f"House {i+1}: ${price * 100000:,.2f}")

# =========================================
# 9. Controlled Experiment (Income Effect)
# =========================================
print("\n===== EFFECT OF MEDIAN INCOME =====")

base_house = [4.0, 30, 5.0, 1.2, 1500, 2.5, 36.0, -119.0]

for income in [2.0, 4.0, 6.0, 8.0]:
    house = base_house.copy()
    house[0] = income

    house_df = pd.DataFrame([house], columns=X.columns)
    prediction = forest.predict(house_df)[0]

    print(f"Income {income:>4}: ${prediction * 100000:,.2f}")

# =========================================
# 10. Interactive Prediction Loop
# =========================================
print("\n===== CUSTOM PREDICTION TOOL =====")

while True:
    print("\nEnter house details (or type 'q' to quit):")

    user_input = input("Median Income: ")
    if user_input.lower() == 'q':
        print("Exiting program. Goodbye!")
        break

    try:
        # Collect all inputs
        medinc = float(user_input)
        houseage = float(input("House Age: "))
        averooms = float(input("Average Rooms: "))
        avebedrms = float(input("Average Bedrooms: "))
        population = float(input("Population: "))
        aveoccup = float(input("Average Occupancy: "))
        latitude = float(input("Latitude: "))
        longitude = float(input("Longitude: "))

        # Create DataFrame for prediction
        user_house = pd.DataFrame([[
            medinc, houseage, averooms, avebedrms,
            population, aveoccup, latitude, longitude
        ]], columns=X.columns)

        # Predict price
        prediction = forest.predict(user_house)[0]

        print(f"\nEstimated House Price: ${prediction * 100000:,.2f}")

    except:
        print("Invalid input. Please enter numeric values.")

