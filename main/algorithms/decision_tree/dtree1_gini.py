""" Decision Tree / Gini Index

The Gini Index is a measure of impurity or diversity within a set of elements.
Gini index is between 0 and 1, it easier to compare gini across different features.
"""

import pandas as pd
import numpy as np

# Dataset
lst = ['apple']*3 + ['orange']*2 + ['banana']*2
fruits = pd.Series(lst)

# Probability distribution
probs = fruits.value_counts(normalize=True)

# Gini Index
gini_index = 1 - np.sum(np.square(probs))

outputs = [
    ["Fruits:", fruits.values],
    ["Probability distribution:", probs.values],
    ["Gini index:", gini_index],
]
for v in outputs: 
    print(v[0], "\n ", v[1])

"""
    Fruits: 
     ['apple' 'apple' 'apple' 'orange' 'orange' 'banana' 'banana']
    Probability distribution: 
     [0.42857143 0.28571429 0.28571429]
    Gini index: 
     0.653061224489796
"""