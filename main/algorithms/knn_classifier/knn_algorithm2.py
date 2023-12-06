import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Train dataset (features and labels)
X = np.array([
    [2, 2, 2], [2, 2.5, 2.5], [2.5, 2.5, 2], [2.5, 2, 2.5], [2.25, 2.25, 2.25],
    [3, 3, 3], [3, 3.5, 3.5], [3.5, 3.5, 3], [3.5, 3, 3.5], [3.25, 3.25, 3.25],
    [4, 4, 4], [4, 4.5, 4.5], [4.5, 4.5, 4], [4.5, 4, 4.5], [4.25, 4.25, 4.25],
])
y = [
    1, 1, 1, 1, 1,
    2, 2, 2, 2, 2,
    3, 3, 3, 3, 3,
]

x_unknown = np.array([3.6, 1.8, 3.6])
k = 3

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2)**2))

# Distances from each training point to the new (unknown) point
D = [euclidean_distance(x, x_unknown) for x in X]
D_nearest = np.argsort(D)[:k]

# Get labels for k-nearest neighbors
knn_targets = [y[i] for i in D_nearest]

# Count occurrences of each label and store in a dictionary
target_counts = {label: knn_targets.count(label) for label in np.unique(knn_targets)}

# Find the target with the maximum occurrence
knn_class = max(target_counts, key=target_counts.get)

# Output prediction
print("Class prediction for", x_unknown, "=", knn_class)

# Plot the point and lines to the k neighbors in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('X3')

# Plot training data points with respective labels
colors = {1: 'r', 2: 'g', 3: 'b'}
for i, label in enumerate(y):
    ax.scatter(X[i, 0], X[i, 1], X[i, 2], c=colors[label])

# Plot unknown point
ax.scatter(x_unknown[0], x_unknown[1], x_unknown[2], marker='x', c='black')

# Plot lines between unknown point and its k-nearest neighbors
for i in D_nearest:
    ax.plot([x_unknown[0], X[i, 0]], [x_unknown[1], X[i, 1]], [x_unknown[2], X[i, 2]], color='gray')

plt.show()
