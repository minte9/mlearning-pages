""" Breast cancer / Logistic Regression

As our data have different values, and even different measurement units, 
we use scalling in order to compare them.
"""

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

# Dataset
dataset = load_breast_cancer()
X = dataset.data
y = dataset.target

# Scalling
X = StandardScaler().fit_transform(X) # Look Here

# Training and test data
X1, X2, y1, y2 = train_test_split(X, y, test_size=0.2, random_state=42)

# Fitting the model
model = LogisticRegression()
model.fit(X1, y1)
score = model.score(X2, y2)

# Predict unknown
X_new = X2[15]
y_pred = model.predict(X_new.reshape(1, -1))
y_expected = dataset['target_names'][y2[15]]
assert y_pred == list(dataset['target_names']).index(y_expected)

# Output
print("Targets:", dataset['target_names'])
print("X_new:\n", X_new)
print("Expected:", y_expected)
print("y_pred:", dataset['target_names'][y_pred][0])

"""
    Targets: ['malignant' 'benign']
    X_new:
    [ 1.41231974  1.62902878  1.52943195  1.35695235  1.7890792   1.41679395
        1.31702506  2.52731642 -0.64847556  1.33855706  0.41082896  3.07195015
        1.45288552  0.58883002  0.16801384  1.95735569 -0.34992981  1.07607669
        1.21292827  2.49460396  0.84092285  1.14687248  1.01387382  0.73378242
        0.29946077  0.17452465 -0.13907293  1.05815376 -0.95409536  0.4479916 ]
    Expected: malignant
    y_pred: malignant
"""