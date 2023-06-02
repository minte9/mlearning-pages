""" Decision Tree / Gini Index

Both entropy and Gini index can be used as impurity measures 
for decision tree algorithms. 

While Gini index is often preferred due to its simplicity and computational efficiency, 
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

# Gini Index
gini_A = 1 - np.sum(np.square(PA))
gini_B = 1 - np.sum(np.square(PB))
assert gini_B < gini_A

outputs = [
    ["A", A.values],
    ["B", B.values],
    ["PA", PA.values],
    ["PB", PB.values],
    ["gini_A", gini_A],
    ["gini_B", gini_B],
]
for v in outputs: 
    print(v[0], v[1])

"""
    Fruits: 
     ['apple' 'apple' 'apple' 'orange' 'orange' 'banana' 'banana']
    Probability distribution: 
     [0.42857143 0.28571429 0.28571429]
    Gini index: 
     0.653061224489796
"""