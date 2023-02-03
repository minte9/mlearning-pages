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


k = 3 # number of k-nearest neighbors to use

xA = np.array([3.6, 1.8]) # unknown class

distances = np.sqrt(np.sum((X - xA)**2, axis=1))
keys = np.argsort(distances)
knn_keys = keys[:k]
knn_classes = y[knn_keys]
knn_most_common_class = np.bincount(knn_classes).argmax()

print("Distances: \n",distances)
print("Keys ordered by distances: \n", keys)
print("Nearest neighbors keys: \n", knn_keys)
print("Nearest neighbors classes: \n", knn_classes)
print("Algorithm class response: \n", knn_most_common_class)

"""
Distances:                  [1.61245155 1.74642492 1.30384048 ...
Keys ordered by distances:  [ 3  8  2  5  4  ...
Nearest neighbors keys:     [3 8 2]
Nearest neighbors classes:  [1 2 1]
Algorithm class response:   1
"""

# -----------------------------------------------
# Plotting

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.set_xlabel('x1')
ax.set_ylabel('x2')

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.scatter(xA[0], xA[1], marker='x', color='r', 
    label='Class =%s' %knn_most_common_class)

for i in knn_keys:
    plt.plot((xA[0], X[i][0]), (xA[1], X[i][1]), color='gray', linestyle='--')

plt.title('K-nearest Neigbors')
plt.xlim(0, 6)
plt.ylim(0, 6)
plt.legend()
plt.show()