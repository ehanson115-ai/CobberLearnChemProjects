# ===============================
# K-Means Clustering on Iris Dataset
# Refined Version
# ===============================

# -------------------------------
# 1. Import Libraries
# -------------------------------
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# 2. Get User Input
# -------------------------------
# Allow user to choose number of clusters
k = int(input("Enter the number of clusters (k): "))

# -------------------------------
# 3. Load Dataset
# -------------------------------
iris = load_iris()

X = iris.data
feature_names = iris.feature_names
y = iris.target
target_names = iris.target_names

# Create DataFrame (includes species for analysis ONLY)
df = pd.DataFrame(X, columns=feature_names)
df['species'] = y

# -------------------------------
# 4. Apply K-Means Clustering
# -------------------------------
kmeans = KMeans(n_clusters=k, random_state=42)
df['cluster'] = kmeans.fit_predict(df[feature_names])

# -------------------------------
# 5. Improved Visualization
# -------------------------------
plt.figure()

scatter = plt.scatter(
    df['petal length (cm)'],
    df['petal width (cm)'],
    c=df['cluster']
)

# Label axes and title clearly
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title(f'K-Means Clustering of Iris Dataset (k = {k})')

# Add colorbar
cbar = plt.colorbar(scatter)
cbar.set_label('Cluster')

# OPTIONAL: plot cluster centers
centers = kmeans.cluster_centers_
plt.scatter(
    centers[:, 2],  # petal length index
    centers[:, 3],  # petal width index
    marker='X',
    s=200
)

# ===============================
# 3D Visualization (Improvement)
# ===============================
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(
    df['sepal length (cm)'],
    df['petal length (cm)'],
    df['petal width (cm)'],
    c=df['cluster']
)

ax.set_xlabel('Sepal Length')
ax.set_ylabel('Petal Length')
ax.set_zlabel('Petal Width')
ax.set_title(f'3D K-Means Clusters (k = {k})')

plt.show()

plt.show()

# -------------------------------
# 6. Analyze Clusters vs Species
# -------------------------------
comparison = pd.crosstab(df['cluster'], df['species'])

print("\nCluster vs Species Count:")
print(comparison)

comparison_percent = comparison.div(comparison.sum(axis=1), axis=0) * 100

print("\nCluster vs Species Percentage:")
print(comparison_percent)