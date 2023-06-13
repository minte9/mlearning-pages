""" Bag of words (weight importance)

We are useing tf-idf (term frequency-inverse document) 
to compare the frequency of a word in a document with the frequency 
of the word in all other documents.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

tweets = [
    "I am eating a burrito, burrito!", 
    "You are eating a pizza"]

# Feature matrix (word imporance)
E = TfidfVectorizer()
B = E.fit_transform(tweets)

# Feature names
F = E.get_feature_names_out()

# Vocabulary
V = E.vocabulary_

# Feature matrix
T = pd.DataFrame(B.toarray(), columns=F)

# ------------------------------------

# Output
print("Tweets:"); print(tweets)
print("Feature matrix (word imporance)"); print(B)
print("Feature matrix (array):"); print(B.toarray())
print("Feature names:"); print(F)
print("Word for each feature:"); print(V)
print("Feature matrix:"); print(T)


"""
    Tweets:
     ['I am eating a burrito, burrito!', 'You are eating a pizza']
    
    Feature matrix (word imporance)
    (0, 2)        0.8523191976057887
    (0, 3)        0.3032160644503863
    (0, 0)        0.42615959880289433
    (1, 4)        0.534046329052269
    (1, 1)        0.534046329052269
    (1, 5)        0.534046329052269
    (1, 3)        0.37997836159100784
    
    Bag of words (array):
     [[0.4261596  0.         0.8523192  0.30321606 0.         0.        ]
      [0.         0.53404633 0.         0.37997836 0.53404633 0.53404633]]
    
    Feature names:
     ['am' 'are' 'burrito' 'eating' 'pizza' 'you']

    Word for each feature:
     {'am': 0, 'eating': 3, 'burrito': 2, 'you': 5, 'are': 1, 'pizza': 4}

    Feature matrix:
            am       are   burrito    eating     pizza       you
    0  0.42616  0.000000  0.852319  0.303216  0.000000  0.000000
    1  0.00000  0.534046  0.000000  0.379978  0.534046  0.534046
"""