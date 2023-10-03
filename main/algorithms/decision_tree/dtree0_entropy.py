""" Decision Tree / Entropy
"""

import pandas as pd
import numpy as np

# Set the initial traning data
A = ['apple']*1 + ['orange']*2 + ['banana']*2
B = ['apple']*5 + ['orange']*2 + ['banana']*0

print("Datasets:")
print("A =", A)
print("B =", B)

# Probability (by hand)
P1 = [1/5, 2/5, 2/5] 
P2 = [5/7, 2/7, 0/7]

print("\n Probability (by hand):")
print("P1 =", P1)
print("P2 =", P2)

# Probability (with pandas)
A = pd.Series(A)
B = pd.Series(B)

P1 = A.value_counts(normalize=True)
P2 = B.value_counts(normalize=True)

print("\n Probability (with pandas):")
print("P1 =", P1.values)
print("P2 =", P2.values)

# Entropy (Shannon model)
H1 = -1 * np.sum(P1 * np.log2(P1))
H2 = -1 * np.sum(P2 * np.log2(P2))
assert H1 > H2

print("\n Entropy:")
print("H1 =", H1)
print("H2 =", H2)

"""
    Datasets:
    A = ['apple' 'orange' 'orange' 'banana' 'banana']
    B = ['apple' 'apple' 'apple' 'apple' 'apple' 'orange' 'orange']

     Probability distributions (by hand):
    [0.2, 0.4, 0.4]
    [0.7142857142857143, 0.2857142857142857, 0.0]

    Probability distributions (with pandas):
    [0.4 0.4 0.2]
    [0.71428571 0.28571429]

    Entropies:
    1.5219280948873621
    0.863120568566631
"""