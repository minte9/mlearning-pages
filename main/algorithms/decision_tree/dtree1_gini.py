""" Decision Tree / Gini Index

Both entropy and Gini index can be used as impurity measures 
for decision tree algorithms. 

Gini index is between 0 and 1, it easy to compare gini across different features.
While Gini index is often preferred due to its computational efficiency, 
entropy may be more sensitive to changes in class probabilities.
"""

import pandas as pd
import numpy as np

# Dataset
A = ['apple']*1 + ['orange']*2 + ['banana']*2
A = pd.Series(A)

B = ['apple']*5 + ['orange']*2 + ['banana']*0
B = pd.Series(B)

# Probability distribution
PA = A.value_counts(normalize=True)
PB = B.value_counts(normalize=True)

# ------------------------------------

# Gini Index
giniA = 1 - np.sum(np.square(PA)) # Look Here
giniB = 1 - np.sum(np.square(PB))
assert giniB < giniA

# ------------------------------------

outputs = [
    ["A  =", A.values],
    ["B  =", B.values],
    ["PA =", PA.values],
    ["PB =", PB.values],
    ["giniA =", giniA],
    ["giniB =", giniB],
]
for v in outputs: 
    print(v[0], v[1])

"""
    A  = ['apple' 'orange' 'orange' 'banana' 'banana']
    B  = ['apple' 'apple' 'apple' 'apple' 'apple' 'orange' 'orange']
    PA = [0.4 0.4 0.2]
    PB = [0.71428571 0.28571429]
    giniA = 0.6399999999999999
    giniB = 0.40816326530612246
"""