""" Decision Tree / Entropy

Entropy tell us how disordered in a collection of data.  
The more impure the feature is, the higher the entropy.
Probability distribution is the frequency of the unique values.
"""

import pandas as pd
import numpy as np

# Dataset
A = ['apple']*3 + ['orange']*2 + ['banana']*2
A = pd.Series(A)

# Probability distribution (by hand or pandas)
P1 = [3/7, 2/7, 2/7]
P2 = A.value_counts(normalize=True)

# Entropy (Shannon model)
E = -1 * np.sum(P1 * np.log2(P1)) # Look Here

# Output
outputs = [
    ["A", A.values],
    ["P1", P1],
    ["P2", P2.values],
    ["E", E],
]
for v in outputs: 
    print(v[0], v[1])

"""
    A  ['apple' 'apple' 'apple' 'orange' 'orange' 'banana' 'banana']
    P1 [0.42857142857142855, 0.2857142857142857, 0.2857142857142857]
    P2 [0.42857143 0.28571429 0.28571429]
    E  1.5566567074628228
"""