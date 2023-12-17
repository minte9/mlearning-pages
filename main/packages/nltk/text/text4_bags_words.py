""" Encoding text (bag of words)

Insteed of storing all values of the matrix, we can store only nonzero values.
One of the nice feature of CountVectorizer is that the output is a 
sparse matrix.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

tweets = [
    "I am eating a burrito, burrito!", 
    "You are eating a pizza"]

# Bag of words features matrix
E = CountVectorizer()
B = E.fit_transform(tweets)

# Feature names
F = E.get_feature_names_out()

# Feature matrix
T = pd.DataFrame(B.toarray(), columns=F)

# ---------------------------------------------------------

# Output
print("Tweets:"); print(tweets)
print("Bag of words (sparce matrix):"); print(B)
print("Bag of words (array):"); print(B.toarray())
print("Feature names:"); print(F)
print("Feature matrix:"); print(T)

"""
    Tweets:
     ['I am eating a burrito', 'You are eating a pizza']

    Bag of words (sparce matrix):
    (0, 0)        1
    (0, 3)        1
    (0, 2)        1
    (1, 3)        1
    (1, 5)        1
    (1, 1)        1
    (1, 4)        1

    Bag of words (array):
     [[1 0 1 1 0 0]
      [0 1 0 1 1 1]]

    Feature names:
     ['am' 'are' 'burrito' 'eating' 'pizza' 'you']

    Feature matrix:
        am  are  burrito  eating  pizza  you
    0   1    0        2       1      0    0
    1   0    1        0       1      1    1
"""