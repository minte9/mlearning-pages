""" Tagging (Tweets)

Convert sentences into features for individual parts of speach.
We use MultiLabelBinarizer encoding.
"""

import nltk
from nltk import pos_tag
from nltk import word_tokenize
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

tweets = [
    "I am eating a burrito", 
    "Political science is amazing",
    "San Francisco is a city"]

def tagging_parts(s):
    A = pos_tag(word_tokenize(s))
    return A

# Tagged tweets
T = []
for tweet in tweets:
    T.append(tagging_parts(tweet))

# Convert tags into features (one-hot encoding)
E = MultiLabelBinarizer()
F = E.fit_transform(T)

# --------------------------------------------------

# Output
print("Tagged tweets:"); print(pd.DataFrame(T))
print("One-hot encoding features:"); print(pd.DataFrame(F))
print("Targets:"); print(pd.DataFrame(E.classes_))

"""
    Tagged tweets:
                    0                 1              2               3              4
    0         (I, PRP)         (am, VBP)  (eating, VBG)         (a, DT)  (burrito, NN)
    1  (Political, JJ)     (science, NN)      (is, VBZ)  (amazing, VBG)           None
    2       (San, NNP)  (Francisco, NNP)      (is, VBZ)         (a, DT)     (city, NN)

    One-hot encoding features:
        0   1   2   3   4   5   6   7   8   9   10  11
    0   0   1   0   0   1   1   0   1   0   1   0   0
    1   0   0   1   0   0   0   1   0   0   0   1   1
    2   1   0   0   1   1   0   0   0   1   0   1   0

    Targets:
    0   (Francisco, NNP)
    1           (I, PRP)
    2    (Political, JJ)
    3         (San, NNP)
    4            (a, DT)
    5          (am, VBP)
    6     (amazing, VBG)
    7      (burrito, NN)
    8         (city, NN)
    9      (eating, VBG)
    10         (is, VBZ)
    11     (science, NN)
"""
