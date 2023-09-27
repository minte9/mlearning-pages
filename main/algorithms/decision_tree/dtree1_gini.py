""" Decision Tree / Gini Index
"""

import pandas as pd
import numpy as np

# Set the initial traning data
A = ['apple']*1 + ['orange']*2 + ['banana']*2
B = ['apple']*5 + ['orange']*2 + ['banana']*0

A = pd.Series(A)
B = pd.Series(B)

# Probability distribution
P1 = A.value_counts(normalize=True)
P2 = B.value_counts(normalize=True)

# Gini Index
g1 = 1 - np.sum(np.square(P1)) # Look Here
g2 = 1 - np.sum(np.square(P2))
assert g1 > g2

# Output results
print("Datasets:")
print("A =", A.values)
print("B =", B.values)

print("\n Probability distributions (by hand):")
print(P1)
print(P2)

print("\n Gini indexes:")
print(g1)
print(g2)

"""
    Datasets:
    A = ['apple' 'orange' 'orange' 'banana' 'banana']
    B = ['apple' 'apple' 'apple' 'apple' 'apple' 'orange' 'orange']

     Probability distributions (by hand):
    orange    0.4
    banana    0.4
    apple     0.2
    dtype: float64
    apple     0.714286
    orange    0.285714
    dtype: float64

     Gini indexes:
    0.6399999999999999
    0.40816326530612246
"""