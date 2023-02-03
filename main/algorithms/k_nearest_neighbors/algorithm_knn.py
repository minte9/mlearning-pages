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

x_unknown = np.array([3.6, 1.8]) # unknown class

distances = np.sqrt(np.sum((X - x_unknown)**2, axis=1))
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
Distances: 
 [1.61245155 1.74642492 1.30384048 1.11803399 1.42302495 1.34164079
 1.80277564 1.70293864 1.20415946 1.49164339 2.23606798 2.72946881
 2.84604989 2.37697286 2.53475837]
Keys ordered by distances: 
 [ 3  8  2  5  4  9  0  7  1  6 10 13 14 11 12]
Nearest neighbors keys: 
 [3 8 2]
Nearest neighbors classes: 
 [1 2 1]
Algorithm class response: 
 1
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