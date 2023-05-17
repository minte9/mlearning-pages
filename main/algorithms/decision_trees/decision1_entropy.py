""" Decisiion Trees / Entropy (Fruits basket)

Probability distribution of the fruits is the reletive frequency of
each fruit in the basket.

The more heterogenous and impure the feature is, the higher the entropy.
We can calculate the impurity of our data using Information gain index.

The more heterogenous and impure the feature is, the higher the Gini index.
Gini index is between 0 and 1, it easier to compare gini across different features.
"""

import pandas as pd
import numpy as np

# ------------------------------------------------------------

# Dataset
lst = ['apple']*3 + ['orange']*2 + ['banana']*2
fruits = pd.Series(lst)

# Relative frequency of each fruit
probs = fruits.value_counts(normalize=True)

# Probability distribution computed by hand
probs2 = [3/7, 2/7, 2/7]

# Entropy (Shannon model)
entropy = -1 * np.sum(probs * np.log2(probs))

# Information gain
gini_index = 1 - np.sum(np.square(probs))

# ------------------------------------------------------------

print("Fruits narray: \n", fruits)
print("Frecvency value_count(): \n", probs)
print("Frecvency bay_hand: \n", probs2)
print("Entropy: \n", entropy)
print("Information gain: \n", gini_index)

"""
    Fruits narray: 
        0     apple
        1     apple
        2     apple
        3    orange
        4    orange
        5    banana
        6    banana

    Frecvency: 
        apple     0.428571
        orange    0.285714
        banana    0.285714

    Frecvency bay_hand: 
        [0.42857142857142855, 0.2857142857142857, 0.2857142857142857]

    Entropy: 
        1.5566567074628228

    Information gain: 
        0.653061224489796
"""