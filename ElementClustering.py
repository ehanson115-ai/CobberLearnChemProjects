import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the CSV
df = pd.read_csv("group1_elements.csv")

# Preview the data
print("Preview of data:")
print(df)
print("\nColumn info:")
print(df.info())

# Select features for clustering
X = df[["atomic_radius_pm", "first_ionization_energy_kJ_per_mol"]]

# Ask user for number of clusters
k = int(input("\nEnter number of clusters (k): "))

# Apply KMeans
kmeans = KMeans(n_clusters=k, random_state=42)
df["cluster"] = kmeans.fit_predict(X)
centers = kmeans.cluster_centers_

# ---- Assign descriptive labels (only meaningful for k=2) ----
cluster_means = df.groupby("cluster")["atomic_radius_pm"].mean()

if k == 2:
    lighter_cluster = cluster_means.idxmin()
    heavier_cluster = cluster_means.idxmax()

    cluster_labels = {
        lighter_cluster: "Lighter Alkali Metals",
        heavier_cluster: "Heavier Alkali Metals"
    }
else:
    cluster_labels = {c: f"Cluster {c}" for c in df["cluster"].unique()}

# Create scatter plot
plt.figure(figsize=(8, 6))

for cluster_id in df["cluster"].unique():
    cluster_data = df[df["cluster"] == cluster_id]
    plt.scatter(
        cluster_data["atomic_radius_pm"],
        cluster_data["first_ionization_energy_kJ_per_mol"],
        s=100,
        edgecolors="black",
        label=cluster_labels[cluster_id]
    )

# Plot cluster centers as smaller, skinnier green Xs
plt.scatter(
    centers[:, 0],
    centers[:, 1],
    marker="x",
    s=150,              # smaller size
    color="green",
    linewidths=1.5,     # thinner lines
    label="Cluster Centers"
)

# Label each element with its symbol
for i, symbol in enumerate(df["symbol"]):
    plt.text(
        df["atomic_radius_pm"][i] + 1,
        df["first_ionization_energy_kJ_per_mol"][i],
        symbol
    )

# Labels and title
plt.xlabel("Atomic Radius (pm)")
plt.ylabel("First Ionization Energy (kJ/mol)")
plt.title(f"K-Means Clustering of Group 1 Elements (k = {k})")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

# Show plot
plt.show()











