""" Decision Tree / Entropy

Entropy tell us how disordered in a collection of data.  
The more impure the feature is, the higher the entropy.
Probability distribution is the frequency of the unique values.
"""

import pandas as pd
import numpy as np

# Dataset
A = ['apple']*1 + ['orange']*2 + ['banana']*2
A = pd.Series(A)

B = ['apple']*5 + ['orange']*2 + ['banana']*0
B = pd.Series(B)

# Probability distribution (by hand or pandas)
P = [3/7, 2/7, 2/7]
PA = A.value_counts(normalize=True)
PB = B.value_counts(normalize=True)

# ------------------------------------

# Entropy (Shannon model)
EA = -1 * np.sum(PA * np.log2(PA))
EB = -1 * np.sum(PB * np.log2(PB))
assert EB < EA

# ------------------------------------

# Output
outputs = [
    ["A  =", A.values],
    ["B  =", B.values],
    ["PA =", PA.values],
    ["PB =", PB.values],
    ["EA =", EA],
    ["EB =", EB],
]
for v in outputs: 
    print(v[0], v[1])

"""
    A  = ['apple' 'orange' 'orange' 'banana' 'banana']
    B  = ['apple' 'apple' 'apple' 'apple' 'apple' 'orange' 'orange']
    PA = [0.4 0.4 0.2]
    PB = [0.71428571 0.28571429]
    EA = 1.5219280948873621
    EB = 0.863120568566631
"""