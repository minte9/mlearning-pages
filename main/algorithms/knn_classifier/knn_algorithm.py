import matplotlib.pyplot as plt
from icecream import ic
import math

# Train dataset (features and labels)
X = [
    [2, 2], [2, 2.5], [2.5, 2.5], [2.5, 2], [2.25, 2.25],
    [3, 3], [3, 3.5], [3.5, 3.5], [3.5, 3], [3.25, 3.25],
    [4, 4], [4, 4.5], [4.5, 4.5], [4.5, 4], [4.25, 4.25],
]
y = [
    1, 1, 1, 1, 1,
    2, 2, 2, 2, 2,
    3, 3, 3, 3, 3,
]

# Unknown (new) data point
x_unknown = [3.6, 1.8]

# Number of nearest neighbors to be used with the algorithm
k = 3

# --------------------------------------------------------------

def euclidean_distance(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    sum = dx**2 + dy**2
    return math.sqrt(sum)

# Distances from each training point to the new (unknown) point
D = []
for x in X:
    distance = euclidean_distance(x[1], x[0], x_unknown[1], x_unknown[0])
    D.append(distance)
ic(D)

# Sort distances (to find the closest points)
D_nearest = sorted(D)
D_nearest = D_nearest[:k]
ic(D_nearest)

# Select k-nearest indices
key_list = []
for i in range(len(D)):
    for d in D_nearest:
        if d == D[i]:
            key_list.append(i)
            
knn_keys = key_list[:k]
ic(knn_keys)

# Get labels for k-nearest neighbors
knn_targets = [y[i] for i in knn_keys]
ic(knn_targets)

# Count occurences of each label and store in dictionary
target_counts = {}
for t in knn_targets:
    if t in target_counts:
        target_counts[t] += 1
    else:
        target_counts[t] = 1

# Find the target with maximum occurence
max_target = max(target_counts, key=target_counts.get)
knn_class = max_target

# Output prediction
print("Class prediction for", x_unknown, "=", knn_class)

# --------------------------------------------------------------

# Plot the point and lines to th k neighbors
fig, ax = plt.subplots()
ax.set_xlabel('x1')
ax.set_ylabel('x2')

# Plot training data points with respective labels
z = x_unknown

# Extract x and y coordinates from X for plotting
x_values = [point[0] for point in X]
y_values = [point[1] for point in X]

plt.scatter(x_values, y_values, c=y)
plt.scatter(z[0], z[1], marker='x', color='r', label='Class =%s' %knn_class)

# Plot dashed lines between unknown point and its k-nearest neighbors
for i in knn_keys:
    plt.plot((z[0], X[i][0]), (z[1], X[i][1]), color='gray', linestyle='--')

plt.title('K-nearest Neigbors')
plt.xlim(0, 6)
plt.ylim(0, 6)
plt.legend()
plt.show()

"""
ic| D: [1.61245154965971,
        1.746424919657298,
        1.3038404810405297,
        1.118033988749895,
        1.4230249470757708,
        1.3416407864998738,
        1.8027756377319946,
        1.70293863659264,
        1.2041594578792296,
        1.4916433890176297,
        2.23606797749979,
        2.7294688127912363,
        2.8460498941515415,
        2.3769728648009427,
        2.534758371127315]
ic| D_nearest: [1.118033988749895, 1.2041594578792296, 1.3038404810405297]
ic| knn_keys: [2, 3, 8]
ic| knn_targets: [1, 1, 2]
Class prediction for [3.6, 1.8] = 1
"""