""" Tagging (parts of speach)

Use NLTK's pre-trained parts-of-speech tager.
NNP proper noun, singular
VBG verb, gerund or present participle
"""

import nltk
from nltk import pos_tag
from nltk import word_tokenize
import pandas as pd

s = "Today science is the technology of tomorrow"

def tagging_parts(s):
    A = pos_tag(word_tokenize(s))
    return A

def print_tags_meanings():
    nltk.help.upenn_tagset()

def find_nouns(A):
    N = [word for word, tag in A \
            if tag in ['NN', 'NNS', 'NNP', 'NNPS']]
    return N

A = tagging_parts(s)
N = find_nouns(A)

# ----------------------------------------------------

# Output
print("Tagging parts:"); print(pd.DataFrame(A))
print("Tag meanings:"); print_tags_meanings()
print("Nouns:"); print(pd.DataFrame(N))

"""
    Tagging parts:
                0    1
    0       Today   NN
    1     science   NN
    2          is  VBZ
    3         the   DT
    4  technology   NN
    5          of   IN
    6    tomorrow   NN

    Tag meanings:
    ...
    NN: noun, common, singular or mass
    NNP: noun, proper, singular
    VBD: verb, past tense
    VBG: verb, present participle or gerund
    ...

            Nouns:
    0       Today
    1     science
    2  technology
    3    tomorrow
"""