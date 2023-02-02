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

X = np.array([[1, 2], [2, 3], [3, 3.5], [5, 5], [5, 6]])
y = np.array([1, 1, 2, 2, 2])

k = 3

new_point = np.array([3, 3])

distances = np.sqrt(np.sum((X - new_point)**2, axis=1))
keys = np.argsort(distances)
k_nearest = keys[:k]

print(distances)    # [2.23606798 1. 0.5 2.82842712 3.60555128]
print(keys)         # [1 2 0 3 4]
print(k_nearest)    # [1 2 0]

k_nearest_classes = y[k_nearest]
most_common_class = np.bincount(k_nearest_classes).argmax()

print("New point class:", most_common_class) # 1