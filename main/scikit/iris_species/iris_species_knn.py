""" Iris Species KNN Clasifier
The most important parameter is the number of neighbors (k)

Our model predicts that this new iris belongs to class 0, 
meaning its species is setosa.
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

dataset = load_iris()
X1, X2, y1, y2 = train_test_split(
    dataset['data'], dataset['target'], random_state=0
)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X1, y1)

X_new = np.array([5, 2.9, 1, 0.2]).reshape(1, 4)
y_new = knn.predict(X_new)
print("Prediction:", y_new)
print("Predicted target:", dataset['target_names'][y_new])
    # [0]
    # setosa