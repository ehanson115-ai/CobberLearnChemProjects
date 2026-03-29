# =========================================
# HOUSING PRICE PREDICTOR GUI (REFINED)
# =========================================

import tkinter as tk
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# =========================================
# 1. Train Model
# =========================================
housing = fetch_california_housing()

df = pd.DataFrame(housing.data, columns=housing.feature_names)
df["MedHouseValue"] = housing.target

X = df.drop("MedHouseValue", axis=1)
y = df["MedHouseValue"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

forest = RandomForestRegressor(random_state=42)
forest.fit(X_train, y_train)

# =========================================
# 2. Create GUI Window
# =========================================
root = tk.Tk()
root.title("California Housing Price Predictor")

# Title
tk.Label(root, text="California Housing Price Predictor",
         font=("Arial", 14)).grid(row=0, column=0, columnspan=2, pady=10)

# =========================================
# 3. Tooltips Helper Class
# =========================================
class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip = None

        widget.bind("<Enter>", self.show)
        widget.bind("<Leave>", self.hide)

    def show(self, event=None):
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + 20

        self.tooltip = tk.Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x}+{y}")

        label = tk.Label(self.tooltip, text=self.text,
                         background="lightyellow", relief="solid", borderwidth=1)
        label.pack()

    def hide(self, event=None):
        if self.tooltip:
            self.tooltip.destroy()

# =========================================
# 4. Input Fields with Defaults + Tooltips
# =========================================
labels = [
    "Median Income",
    "House Age",
    "Average Rooms",
    "Average Bedrooms",
    "Population",
    "Average Occupancy",
    "Latitude",
    "Longitude"
]

defaults = ["4.0", "30", "5.0", "1.2", "1500", "2.5", "36.0", "-119.0"]

tooltips = [
    "Average income in the area",
    "Average age of houses",
    "Average number of rooms",
    "Average number of bedrooms",
    "Population of the area",
    "Average number of people per house",
    "Geographic latitude",
    "Geographic longitude"
]

entries = []

for i, label in enumerate(labels):
    lbl = tk.Label(root, text=label)
    lbl.grid(row=i+1, column=0, padx=10, pady=5, sticky="w")

    entry = tk.Entry(root)
    entry.grid(row=i+1, column=1, padx=10, pady=5)

    entry.insert(0, defaults[i])  # Default value
    ToolTip(entry, tooltips[i])   # Tooltip

    entries.append(entry)

# =========================================
# 5. Prediction Function (Improved Validation)
# =========================================
def predict_price():
    values = []

    try:
        for entry in entries:
            value = entry.get().strip()

            if value == "":
                raise ValueError("Empty field")

            values.append(float(value))

        input_df = pd.DataFrame([values], columns=X.columns)
        prediction = forest.predict(input_df)[0]

        price = prediction * 100000

        result_label.config(
            text=f"Predicted Price: ${price:,.2f}",
            fg="green"
        )

    except:
        result_label.config(
            text="❌ Please enter valid numeric values in all fields.",
            fg="red"
        )

# =========================================
# 6. Clear Button
# =========================================
def clear_fields():
    for i, entry in enumerate(entries):
        entry.delete(0, tk.END)
        entry.insert(0, defaults[i])

    result_label.config(text="Predicted Price: ", fg="black")

# =========================================
# 7. Buttons
# =========================================
tk.Button(root, text="Predict Price", command=predict_price)\
    .grid(row=10, column=0, pady=10)

tk.Button(root, text="Clear", command=clear_fields)\
    .grid(row=10, column=1, pady=10)

# =========================================
# 8. Output Label
# =========================================
result_label = tk.Label(root, text="Predicted Price: ")
result_label.grid(row=11, column=0, columnspan=2, pady=10)

# =========================================
# 9. Run GUI
# =========================================
root.mainloop()