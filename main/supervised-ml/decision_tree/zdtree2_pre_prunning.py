""" Decision Trees / Pre-prunning

Stop developing the tree before the limit of perfectly train data fit.
This will lower the accuracy for train data, but it will improve 
the score on test data.
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Dataset
dataset = load_breast_cancer()

# Training and test data
X1, X2, y1, y2 = train_test_split(
    dataset.data, dataset.target, stratify=dataset.target, random_state=42)

# Pre-prunning
tree = DecisionTreeClassifier(max_depth=4, random_state=0)
tree.fit(X1, y1)

# Accuracy
score_train = tree.score(X1, y1)
score_test = tree.score(X2, y2)

print(f"Accuracy on training set: {score_train:.3f}")
print(f"Accuracy on test set: {score_test:.3f}")

"""
    Accuracy on training set: 0.988
    Accuracy on test set: 0.951
"""