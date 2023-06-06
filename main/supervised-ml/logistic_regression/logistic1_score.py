""" Logistic Regression (Examp Scores)

Logistic Regression transforms the linear combination of the input features 
into a range of probabilites.

Linear Regression predicts continuous numeric values, while 
Logistic Regression predicts probabilities for categorical outcomes.
If we can't rely on similarity to make predictions, as with KNN Classifer, 
Logistic Regression is a better choice.

Suppose we have two features: exam1 score (continuous) and exam2 score (continuous). 
The target variable is student admission (binary: 0/1 not admited or admited).
"""

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Dataset
X = [[80, 85], [90, 95], [70, 75], [60, 65]]
y = [1, 1, 0, 0]

# Train and test data
X1, X2, y1, y2 = train_test_split(X, y, test_size=0.2, random_state=42)

# Fitting the model
model = LogisticRegression()
model.fit(X1, y1)
score = model.score(X2, y2)

# Prediction
x_new =  [92, 64]
y_pred = model.predict([x_new])
assert y_pred == 1

print("Unknown:", x_new)
print("Prediction:", y_pred)
print("Score:", round(score,2))

"""
    Unknown: [92, 64]
    Prediction: [1]
    Score: 1.0
"""