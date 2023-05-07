""" K-Nearest Neighbors Classifier
"""

import numpy as np
import matplotlib.pyplot as plt

# Train dataset
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

class KNeighborsClassifier:

    def __init__(self, n_neighbors):
        self.k = n_neighbors

    def fit(self, X_train, y_train):
        self.X = np.array(X_train)
        self.y = np.array(y_train)
    
    def predict(self, x_unknown):
        X = self.X
        y = self.y
        k = self.k
        z = np.array(x_unknown)

        SD = np.sqrt(np.sum((X - z)**2, axis=1))
        keys = np.argsort(SD)
        keys_knn = keys[:k]
        targets_knn = y[keys_knn]
        most_common = np.bincount(targets_knn)
        result = most_common.argmax()
        
        return result

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

x_unknown = [3.6, 1.8]  
knn_class = knn.predict(x_unknown)   
print("Prediction for", x_unknown, "= class", knn_class)

"""
    Square distances:           [1.61245155 1.74642492 1.30384048 ...]
    Keys ordered by distances:  [ 3  8  2  5  4  9  0  7 ...]
    Nearest neighbors keys:     [3 8 2]
    Nearest neighbors targets:  [1 2 1]
    Algorithm target response:  1
    Class prediction:           [3.6, 1.8] = 1
"""