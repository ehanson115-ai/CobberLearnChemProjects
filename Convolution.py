import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

# -------------------------
# Create the 5x5 image
# -------------------------
image = np.array([
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [1, 1, 1, 1, 1],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0]
])

print("Image array:")
print(image)
print("Shape:", image.shape)


# -------------------------
# Create horizontal edge filter
# -------------------------
filter_kernel = np.array([
    [ 1,  1,  1],
    [ 0,  0,  0],
    [-1, -1, -1]
])

print("\nFilter:")
print(filter_kernel)


# -------------------------
# Apply convolution
# -------------------------
feature_map = convolve2d(image, filter_kernel, mode='valid')

print("\nFeature Map:")
print(feature_map)
print("Feature map shape:", feature_map.shape)


# -------------------------
# Visualize input + feature map
# -------------------------
plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.title("Input Image")
plt.imshow(image)
plt.colorbar()

plt.subplot(1,2,2)
plt.title("Feature Map")
plt.imshow(feature_map)
plt.colorbar()

plt.tight_layout()
plt.show()


# -------------------------
# Roman numeral examples
# -------------------------
numerals = {

    "I": np.array([
        [0,1,0],
        [0,1,0],
        [0,1,0],
        [0,1,0],
        [0,1,0]
    ]),

    "V": np.array([
        [1,0,0,0,1],
        [1,0,0,0,1],
        [0,1,0,1,0],
        [0,1,0,1,0],
        [0,0,1,0,0]
    ]),

    "X": np.array([
        [1,0,0,0,1],
        [0,1,0,1,0],
        [0,0,1,0,0],
        [0,1,0,1,0],
        [1,0,0,0,1]
    ])
}


# -------------------------
# Apply filter to numerals
# -------------------------
for name, img in numerals.items():

    fmap = convolve2d(img, filter_kernel, mode='valid')

    plt.figure(figsize=(8,3))

    plt.subplot(1,2,1)
    plt.title(f"Roman Numeral {name}")
    plt.imshow(img)
    plt.colorbar()

    plt.subplot(1,2,2)
    plt.title("Feature Map")
    plt.imshow(fmap)
    plt.colorbar()

    plt.tight_layout()
    plt.show()

    # Sample Output:
    # Image array:
    # [[0 0 1 0 0]
    #  [0 0 1 0 0]
    #  [1 1 1 1 1]
    #  [0 0 1 0 0]
    #  [0 0 1 0 0]]
    # Shape: (5, 5)
    #
    # Filter:
    # [[ 1  1  1]
    #  [ 0  0  0]
    #  [-1 -1 -1]]
    #
    # Feature Map:
    # [[ 0  1  0]
    #  [-3 -2 -3]
    #  [ 0  1  0]]
    # Feature map shape: (3, 3)