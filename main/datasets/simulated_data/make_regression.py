""" Make regression
Dataset designed to be used with linear regression.
"""

import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

# Sample dataset
features, target, coef = make_regression(
    n_samples = 100,
    n_features = 3,
    n_informative = 3,
    n_targets = 1,
    noise = 0,
    coef = True,
    random_state = 1
)

# Feature Matrix (first three rows)
print(features[:2])
    # [[ 1.29322588 -0.61736206 -0.11044703]
    #  [-2.793085    0.36633201  1.93752881]
    #  [ 0.80186103 -0.18656977  0.0465673 ]]

# Target vector (first three elements)
print(target[:3])
    # [-10.37865986  25.5124503   19.67705609]
