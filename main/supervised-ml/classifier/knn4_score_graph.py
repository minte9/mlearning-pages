"""KNN plot score
Models between k=3 and k=7 perform optimally on the test set
They optimally balance overfitting and underfitting
"""
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import matplotlib.pyplot as plt

A = pd.DataFrame({
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

B = pd.DataFrame({
    'height': [4, 4.47, 6.49, 7.51, 8.34],
    'width': [6.5, 7.13, 7, 5.01, 4.23],
    'fruit': ['Mandarin', 'Mandarin', 'Apple', 'Lemon', 'Lemon']
})

X  = A[['height', 'width']].values
X2 = B[['height', 'width']].values

y  = A.fruit.values
y2 = B.fruit.values

k = []
score = []
score2 = []

for i in range(len(X)):
    _k = i+1
    
    clf = KNeighborsClassifier(n_neighbors = _k)
    clf.fit(X, y)

    _score = metrics.accuracy_score(y, clf.predict(X))
    _score2 = metrics.accuracy_score(y2, clf.predict(X2))

    k.append(_k)
    score.append(_score * 100)
    score2.append(_score2 * 100)
    
    print(f'k={_k} | score: {score[i]} | score2: {score2[i]}')


# Plot train score
plt.scatter(k, score) #function
plt.plot(k, score, '-', label='train') #data points

# Plot test score
plt.scatter(k, score2)
plt.plot(k, score2, '-', label='test')

# Plot configurations
plt.axis([max(k),min(k)+1, 0, 100])
plt.xlabel('number of nearest neighbours (k)', size = 13)
plt.ylabel('accuracy score', size = 13)
plt.title('Model Performance vs Complexity', size = 20)
plt.legend()

# Output
plt.show()