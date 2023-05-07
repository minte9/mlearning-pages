""" K-Nearest Neighbors Classifier / Algorithm

The algorithm works by finding the k nearest neighbors to the input data point, 
and then assigning the output label based on the majority 
vote of the k-nearest neighbors.

In classical algorithms, the f(x) is provided by the programmer.
In ML, the computer uses data to learn the best f(x) fit.

1. Load training dataset
2. Select a value for k
3. Calculate distances to the new point
4. Select the k-nearest points
5. Get the most common class
6. Assign the new point to that class
"""

import numpy as np
import matplotlib.pyplot as plt

# Train dataset
X = np.array([
    [2, 2], [2, 2.5], [2.5, 2.5], [2.5, 2], [2.25, 2.25],
    [3, 3], [3, 3.5], [3.5, 3.5], [3.5, 3], [3.25, 3.25],
    [4, 4], [4, 4.5], [4.5, 4.5], [4.5, 4], [4.25, 4.25],
])
y = np.array([
    1, 1, 1, 1, 1,
    2, 2, 2, 2, 2,
    3, 3, 3, 3, 3,
])

k_nearest = 3
x_unknown = np.array([3.6, 1.8])

# Algorithm
# --------------------------------------------------------------

# Square distances matrix
SD = np.sqrt(np.sum((X - x_unknown)**2, axis=1)) # axis=1 means rows of X
keys = np.argsort(SD)

# Neighbors target matrix
keys_knn = keys[:k_nearest]
targets_knn = y[keys_knn]

# Optim target
most_common = np.bincount(targets_knn) # by number of occurrences 
knn_class = most_common.argmax() # max value in array

# --------------------------------------------------------------

# Plot the point and lines to th k neighbors
fig, ax = plt.subplots()
ax.set_xlabel('x1')
ax.set_ylabel('x2')

z = x_unknown
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.scatter(z[0], z[1], marker='x', color='r', label='Class =%s' %knn_class)

for i in keys_knn:
    plt.plot((z[0], X[i][0]), (z[1], X[i][1]), color='gray', linestyle='--')

plt.title('K-nearest Neigbors')
plt.xlim(0, 6)
plt.ylim(0, 6)
plt.legend()
plt.show()

print("Square distances:", SD)
print("Keys ordered by distances:", keys)
print("Nearest neighbors keys:", keys_knn)
print("Nearest neighbors targets:", targets_knn)
print("Algorithm target response:", knn_class)
print("Class prediction for", x_unknown, "=", knn_class)

"""
    Square distances:           [1.61245155 1.74642492 1.30384048 ...]
    Keys ordered by distances:  [ 3  8  2  5  4  9  0  7 ...]
    Nearest neighbors keys:     [3 8 2]
    Nearest neighbors targets:  [1 2 1]
    Algorithm target response:  1
    Class prediction:           [3.6, 1.8] = 1
"""