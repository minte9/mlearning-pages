"""KNN Evaluation
Evaluate the model on the training and test dataset
The score is the difference between actual and predicted labels
1.0 means the model correctly predicted all (100%)
"""

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

A = pd.DataFrame({ # training dataset
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

B = pd.DataFrame({   # test dataset
    'height': [4, 4.47, 6.49, 7.51, 8.34],
    'width':  [6.5, 7.13, 7, 5.01, 4.23],
    'fruit':  ['Mandarin', 'Mandarin', 'Apple', 'Lemon', 'Lemon']
})

X  = A[['height', 'width']].values
X2 = B[['height', 'width']].values
y  = A.fruit.values
y2 = B.fruit.values

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

predictions = knn.predict(X)
score = metrics.accuracy_score(y, predictions) # evaluate on training dataset
print(score * 100) # 85%

predictions = knn.predict(X2)
score = metrics.accuracy_score(y2, predictions) # evaluate on test dataset
print(score * 100) # 100%