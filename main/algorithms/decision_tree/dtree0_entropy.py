import pandas as pd
import numpy as np
from icecream import ic

# Set the initial traning data
A = ['apple']*1 + ['orange']*2 + ['banana']*2
B = ['apple']*5 + ['orange']*2 + ['banana']*0
ic(A, B)

# Probability
P1 = pd.Series(A).value_counts(normalize=True)
P2 = pd.Series(B).value_counts(normalize=True)
ic(P1, P2)

# Entropy (Shannon model)
P1 = P1.values
P2 = P2.values
H1 = -1 * np.sum(P1 * np.log2(P1))
H2 = -1 * np.sum(P2 * np.log2(P2))
ic(H1, H2);

assert H1 > H2

ic("A entropy > B entropy | There is more disorder in A than B")
ic("Assertion passed");

"""
    ic| A: ['apple', 'orange', 'orange', 'banana', 'banana']
        B: ['apple', 'apple', 'apple', 'apple', 'apple', 'orange', 'orange']
    ic| P1: orange    0.4
            banana    0.4
            apple     0.2
            dtype: float64
        P2: apple     0.714286
            orange    0.285714
            dtype: float64
    ic| H1: 1.5219280948873621, H2: 0.863120568566631
    ic| 'A entropy > B entropy | There is more disorder in A than B'
    ic| 'Assertion passed'
"""