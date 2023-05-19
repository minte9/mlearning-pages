""" Decision Trees / Prunning

Insteed of looking at the whole tree, we can select only the most useful properties.
To summarize the feature importance we use `tree.feature_importance`.
We can see that `worst radius` used in the top split, is by far 
the most important feature.
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

import pathlib
DIR = pathlib.Path(__file__).resolve().parent

# Dataset
dataset = load_breast_cancer()

# Training and test data
X1, X2, y1, y2 = train_test_split(
    dataset.data, dataset.target, stratify=dataset.target, random_state=42)

# Pre-prunning
dtree = DecisionTreeClassifier(max_depth=4, random_state=0)
dtree.fit(X1, y1)

# Output
n = dataset.data.shape[1]
plt.subplots_adjust(left=0.28)
plt.barh(np.arange(n), dtree.feature_importances_, align='center')
plt.yticks(np.arange(n), dataset.feature_names)
plt.xlabel("Feature importance")
plt.ylabel("Feature")
plt.ylim(-1, n)
plt.show()

print('Feature importances: \n', dtree.feature_importances_)

"""
    Feature importances: 
    [0.         0.         0.         0.         0.         0.
    0.         0.         0.         0.         0.01019737 0.04839825
    0.         0.         0.0024156  0.         0.         0.
    0.         0.         0.72682851 0.0458159  0.         0.
    0.0141577  0.         0.018188   0.1221132  0.01188548 0.        ]
"""
