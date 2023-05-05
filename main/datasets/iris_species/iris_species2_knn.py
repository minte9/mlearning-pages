""" Iris Species / KNN Clasifier

Learn model that predicts the species of a new iris
based on known measurements (length and width of petals).
The most important parameter is the number of neighbors (k)

Our model predicts that this new iris belongs to class 0, 
meaning its species is setosa.
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import matplotlib.pyplot as plt

# Training and test datasets
dataset = load_iris()
X1, X2, y1, y2 = train_test_split(
    dataset['data'], dataset['target'], random_state=0
)

# Learn model
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X1, y1)

# New iris prediction
X_new = np.array([5, 2.9, 1, 0.2]).reshape(1, 4)
y_new = knn.predict(X_new)

# Training data frame
df = pd.DataFrame(X1, columns=dataset.feature_names)

# Plot iris petal
fig, ax = plt.subplots()
ax.set_title("Petals")
ax.set_xlabel('length (cm)')
ax.set_ylabel('width (cm)')
ax.scatter(df['petal length (cm)'], df['petal width (cm)'], c=y1)
ax.scatter(X_new[0][2], X_new[0][3], c='r', marker='x', s=100)
ax.grid()

# Plot iris sepal
fig, ax = plt.subplots()
ax.set_title("Sepals")
ax.set_xlabel('length (cm)')
ax.set_ylabel('width (cm)')
ax.scatter(df['sepal length (cm)'], df['sepal width (cm)'], c=y1)
ax.scatter(X_new[0][0], X_new[0][1], c='r', marker='x', s=100)
ax.grid()

# Plot the new point on the scatter matrix plot
axes = pd.plotting.scatter_matrix(
    df, c=y1, figsize=(15, 15), marker='o', 
    s=60, alpha = .8, diagonal='none'
)
for i in range(4):
    for j in range(4):
        if i == j:
            continue
        ax = axes[i, j]
        ax.scatter(X_new[:, j], X_new[:, i], c='r', marker='x', s=200)

plt.show()

print("Prediction class:", y_new)
print("Predicted target:", dataset['target_names'][y_new])

"""
    Prediction class: [0]
    Predicted target: ['setosa']
"""