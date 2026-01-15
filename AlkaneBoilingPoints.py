import matplotlib.pyplot as plt
import os

# Number of carbons in the first 10 linear alkanes
num_carbons = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Corresponding boiling points in Celsius
boiling_points = [-161.5, -88.6, -42.1, -0.5, 36.1, 68.7, 98.4, 125.6, 150.8, 174.1]

# Directory and filename
save_dir = "AlkaneBoilingPoints"
file_name = "boiling_point_vs_carbons.png"

# Create directory if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

# Create a new figure
plt.figure()

# Set a very light blue background
plt.gca().set_facecolor('#e6f2ff')

# Scatter plot with pink star markers
plt.scatter(num_carbons, boiling_points, color='pink', marker='*', s=150)

# Add title and labels
plt.title("Boiling Point vs. Number of Carbons in Linear Alkanes")
plt.xlabel("Number of Carbons")
plt.ylabel("Boiling Point (°C)")

# Save the figure
plt.savefig(os.path.join(save_dir, file_name), dpi=300, bbox_inches='tight')

# Display the plot
plt.show(block=True)


