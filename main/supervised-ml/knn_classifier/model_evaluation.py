# MODEL EVALUATION - KNN
# ----------------------
# The dataset is splitted in two datasets (training and test) that
# will be used to evaluate the model.
# The score is the difference between actual and predicted labels.
# A score of 1.0 means that the model correctly predicted all labels (100%).
# ------------------------------------------------

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

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

# Train the model
# ---------------
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X1, y1)


# Evaluate the model
# ------------------
predictions1 = knn.predict(X1)
predictions2 = knn.predict(X2)

score1 = metrics.accuracy_score(y1, predictions1)
score2 = metrics.accuracy_score(y2, predictions2)

print("Model score on training dataset:", score1 * 100)
print("Model score on test dataset:", score2 * 100)

# -----------------------------------
# Model score on training dataset: 85.0
# Model score on test dataset: 100.0