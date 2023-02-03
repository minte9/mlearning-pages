""" K-Nearest Neighbors Algorithm

In ML, the computer uses data to learn the best f(x) 
In classical algorithms, the f(x) is provided by the programmer

1. Load training dataset
2. Select a value for k
3. Calculate distances to the new point
4. Select the k-nearest points
5. Get the most common class
6. Assign the new point to that class
"""

import numpy as np

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


k = 3 # number of k-nearest neigbours to use

x_unknown = np.array([3.4, 2.2]) # unknown class

distances = np.sqrt(np.sum((X - x_unknown)**2, axis=1))
keys = np.argsort(distances)
knn_keys = keys[:k]
knn_classes = y[knn_keys]
knn_most_common_class = np.bincount(knn_classes).argmax()

print(distances)
print(keys)
print(knn_keys)
print(knn_classes)
print("New point class:", knn_most_common_class)

"""
    [1.58113883 1.5        1.27475488 1.         1.58113883 1.25
    0.70710678 1.11803399 0.79056942 1.         1.58113883 1.03077641
    1.58113883 2.06155281 1.90394328 2.23606798 2.54950976 2.13600094]

    [ 6  3  8  5  2 11  7  9  1  0  4 10 12 14 13 17 15 16]
    [ 6  8  3  9 11]
    [2 2 1 2 2]

    New point class: 2
"""

# -----------------------------------------------
# Plotting

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.set_xlabel('x1')
ax.set_ylabel('x2')

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.scatter(x_unknown[0], x_unknown[1], marker='x', color='r')

for i in knn_keys:
    plt.plot((x_unknown[0], X[i][0]), (x_unknown[1], X[i][1]), 
        color='gray', linestyle='--')

plt.xlim(0, 6)
plt.ylim(0, 6)
plt.colorbar
plt.show()