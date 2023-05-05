""" Iris species / Model Evaluation

We make a prediction for each iris in the test dataset and
compare it against its known label.

For this model the test set accurary is 0.97
This means that we made the right prediction for 97% 
for irises in the test dataset.
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

dataset = load_iris()
X1, X2, y1, y2 = train_test_split(
    dataset['data'], dataset['target'], random_state=0
)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X1, y1)

y_new = knn.predict(X2)         # predictions on test dataset
score1 = np.mean(y_new == y2)   # get score using average, OR
score2 = knn.score(X2, y2)      # knn object

print("Score using mean():", round(score1, 2))
print("Score using knn object:", round(score2, 2))

"""
    Score using mean(): 0.97
    Score using knn object: 0.97
"""