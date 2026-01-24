# KNN - SCORE GRAPH
# -----------------
# By plotting the results graph we can see that the model 
# perform optimally for k between 3 and 7 (for test set).
# ------------------------------------------------

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import matplotlib.pyplot as plt

# Training and TEST datasets
# --------------------------
D1 = pd.DataFrame({
  'height': [
    3.91, 7.09, 10.48, 9.21, 7.95, 7.62, 7.95, 4.69, 7.50, 7.11, 
    4.15, 7.29, 8.49, 7.44, 7.86, 3.93, 4.40, 5.5, 8.10, 8.69
  ], 
  'width': [
     5.76, 7.69, 7.32, 7.20, 5.90, 7.51, 5.32, 6.19, 5.99, 7.02, 
     5.60, 8.38, 6.52, 7.89, 7.60, 6.12, 5.90, 4.5, 6.15, 5.82
  ],
  'fruit': [
    'Mandarin', 'Apple', 'Lemon', 'Lemon', 'Lemon', 'Apple', 'Mandarin', 
    'Mandarin', 'Lemon', 'Apple', 'Mandarin', 'Apple', 'Lemon', 'Apple', 
    'Apple', 'Apple', 'Mandarin', 'Lemon', 'Lemon', 'Lemon'
  ]
})

# Test dataset
D2 = pd.DataFrame({
    'height': [4, 4.47, 6.49, 7.51, 8.34],
    'width':  [6.5, 7.13, 7, 5.01, 4.23],
    'fruit':  ['Mandarin', 'Mandarin', 'Apple', 'Lemon', 'Lemon']
})


# Transform datasets
# ------------------
X1 = D1[['height', 'width']].values
y1 = D1.fruit.values

X2 = D2[['height', 'width']].values
y2 = D2.fruit.values


# Evaluate the score for different params
# ---------------------------------------
k = []
score1 = []
score2 = []

for i in range(len(X1)):
    _k = i + 1  # different k neighbors

    knn = KNeighborsClassifier(n_neighbors=_k)
    knn.fit(X1, y1)

    _score1 = metrics.accuracy_score(y1, knn.predict(X1))
    _score2 = metrics.accuracy_score(y2, knn.predict(X2))  # test dataset

    k.append(_k)
    score1.append(_score1 * 100)
    score2.append(_score2 * 100)

    # Output accuracy for each _k (n_neighbors)
    print(f"k={_k} | score1: {score1[i]} | score2: {score2[i]}")

    """
    k=1 | score1: 100.0 | score2: 40.0
    k=2 | score1: 95.0 | score2: 60.0
    k=3 | score1: 85.0 | score2: 100.0
    k=4 | score1: 85.0 | score2: 100.0
    k=5 | score1: 85.0 | score2: 100.0
    k=6 | score1: 85.0 | score2: 100.0
    k=7 | score1: 85.0 | score2: 100.0
    k=8 | score1: 85.0 | score2: 100.0
    k=9 | score1: 85.0 | score2: 80.0
    k=10 | score1: 85.0 | score2: 60.0
    k=11 | score1: 80.0 | score2: 60.0
    k=12 | score1: 90.0 | score2: 60.0
    k=13 | score1: 65.0 | score2: 60.0
    k=14 | score1: 55.00000000000001 | score2: 60.0
    k=15 | score1: 55.00000000000001 | score2: 60.0
    k=16 | score1: 45.0 | score2: 60.0
    k=17 | score1: 50.0 | score2: 60.0
    k=18 | score1: 50.0 | score2: 60.0
    k=19 | score1: 40.0 | score2: 40.0
    k=20 | score1: 40.0 | score2: 40.0
    """


# Plot results
# ------------

# Plot train score
plt.scatter(k, score1)  # function
plt.plot(k, score1, '-', label='train')  # data points

# Plot test score
plt.scatter(k, score2)  # function
plt.plot(k, score2, '-', label='test')  # data points

# Plot configurations
plt.axis([max(k),min(k)+1, 0, 100])
plt.xlabel('number of nearest neighbours (k)', size = 13)
plt.ylabel('accuracy score', size = 13)
plt.title('Model Performance vs Complexity', size = 20)
plt.legend()

# Output
plt.show()
