import numpy as np
import matplotlib.pyplot as plt

# Train datasets
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
        z = np.array(x_unknown)
        
        # Square distances
        SD = np.sqrt(np.sum((self.X - z)**2, axis=1))
        keys = np.argsort(SD)

        # Neighbors targets
        keys_knn = keys[:self.k]
        targets_knn = self.y[keys_knn]

        # Optim target
        most_common = np.bincount(targets_knn)
        result = most_common.argmax()
        
        return result

# Instantiate KNeighborsClassifier with 3 neighbors
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

# Unknown data point
x_unknown = [3.6, 1.8]  
knn_class = knn.predict(x_unknown)

# Output prediction
print("Prediction for", x_unknown, "= class", knn_class)

"""
    Prediction for [3.6, 1.8] = class 1
"""