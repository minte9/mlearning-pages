""" Decision Tree / Entropy

Entropy is a measure of how disordered a collection of data is.  
The more heterogenous and impure the feature is, the higher the entropy.
"""

import pandas as pd
import numpy as np

# Dataset
lst = ['apple']*3 + ['orange']*2 + ['banana']*2
fruits = pd.Series(lst)

# Probability distribution (relative frequency of fruits)
probs = fruits.value_counts(normalize=True)
probs2 = [3/7, 2/7, 2/7] # by hand

# Entropy (Shannon model)
entropy = -1 * np.sum(probs * np.log2(probs))

outputs = [
    ["Fruits:", fruits.values],
    ["Probability distribution - value_count():", probs.values],
    ["Probability distribution - by hand:", probs2],
    ["Entropy:", entropy],
]
for v in outputs: 
    print(v[0], "\n ", v[1])

"""
    Fruits: 
     ['apple' 'apple' 'apple' 'orange' 'orange' 'banana' 'banana']
    Probability distribution - value_count(): 
     [0.42857143 0.28571429 0.28571429]
    Probability distribution - by hand: 
     [0.42857142857142855, 0.2857142857142857, 0.2857142857142857]
    Entropy: 
     1.5566567074628228
"""