""" K-Nearest Neighbors Algorithm

In ML, the computer uses data to learn the best f(x) fit
In classical algorithms, the f(x) is provided by the programmer

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

        # Algorithm
        # --------------------------------------------------------------

        # Square distances matrix
        SD = np.sqrt(np.sum((X - z)**2, axis=1)) # axis=1 means rows of X
        keys = np.argsort(SD)

        # Neighbors target matrix
        keys_knn = keys[:k]
        targets_knn = y[keys_knn]

        # Optim target
        most_common = np.bincount(targets_knn) # by number of occurrences 
        result = most_common.argmax() # max value in array

        # --------------------------------------------------------------

        print("Square distances: \n", SD)
        print("Keys ordered by distances: \n", keys)
        print("Nearest neighbors keys: \n", keys_knn)
        print("Nearest neighbors targets: \n", targets_knn)
        print("Algorithm target response: \n", result)

        # Plot the point and lines to th k neighbors
        fig, ax = plt.subplots()
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')

        plt.scatter(X[:, 0], X[:, 1], c=y)
        plt.scatter(z[0], z[1], marker='x', color='r', label='Class =%s' %result)

        for i in keys_knn:
            plt.plot((z[0], X[i][0]), (z[1], X[i][1]), color='gray', linestyle='--')

        plt.title('K-nearest Neigbors')
        plt.xlim(0, 6)
        plt.ylim(0, 6)
        plt.legend()
        plt.show()

        return result

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

x_unknown = [3.6, 1.8]  
knn_class = knn.predict(x_unknown)   
print("Class prediction for", x_unknown, "=", knn_class)

"""
    Square distances: 
    [1.61245155 1.74642492 1.30384048 1.11803399 ...]
    
    Keys ordered by distances: 
    [ 3  8  2  5  4  9  0  7  1  6 10 13 14 11 12]
    
    Nearest neighbors keys: 
    [3 8 2]
    
    Nearest neighbors targets: 
    [1 2 1]
    
    Algorithm target response: 
    1
    
    Class prediction for [3.6, 1.8] = 1
"""