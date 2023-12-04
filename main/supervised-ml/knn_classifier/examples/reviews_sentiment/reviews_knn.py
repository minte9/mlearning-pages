""" Reviews Sentiment (KNN)

Collect reviews from amazon books and manually label each post as positive or negative sentiment.
Extract text features from the posts, such as word frequencies or TF-IDF values.
Build a dataset with these features and labels.
Train a KNN classifier to classify new posts based on their sentiment.
"""

import pathlib
import pandas as pd
import numpy as np
import sys
import unicodedata
import re
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from icecream import ic

# Dataset
DIR = pathlib.Path(__file__).resolve().parent
df = pd.read_csv(DIR / 'reviews.csv')

def clear_text(A):
    A = [s.strip() for s in A]
    A = [re.sub(r"[0-9]", "", s) for s in A]
    A = [s.lower() for s in A]
    return A

def remove_punctuation(A):
    P = dict()
    for i in range(sys.maxunicode):
        if unicodedata.category(chr(i)).startswith('P'):
            P[i] = None
    A = [s.translate(P) for s in A]
    return A

def tokenize_words(A):
    A = [word_tokenize(s) for s in A]
    return A

def remove_stopwords(A):
    A = [[w for w in words if w not in stopwords.words('english')] for words in A]
    return A

def steming_words(A):
    porter = PorterStemmer()
    A = [[porter.stem(w) for w in words] for words in A]
    return A

def debug(s1, s2=None, debug=True):
    if debug:
        print(s1, s2)

ic("Initial dataset")
ic(df.head())

ic("Clear Text (remove numbers, lower, strip)")
df['Review'] = clear_text(df['Review'])
ic(df.head())

ic("Remove Punctuation")
df['Review'] = remove_punctuation(df['Review'])
ic(df.head())

ic("Tokenize Words")
df['Words']  = tokenize_words(df['Review'])
ic(df['Words'].head())

ic("Remove Stop Words")
df['Words']  = remove_stopwords(df['Words'])
ic(df['Words'].head())

ic("Stem Words")
df['Words']  = steming_words(df['Words'])
id(df['Words'].head())

def build_dataset(df):
    # E = CountVectorizer() # Bag of words
    E = TfidfVectorizer() # Word importance

    # Features matrix
    F = E.fit_transform([" ".join(words) for words in df['Words']])
    T = pd.DataFrame(F.toarray(), columns=E.get_feature_names_out())

    # Dataset
    dataset = pd.concat([T, df['Sentiment']], axis=1)
    return dataset

id("Final dataset")
dataset = build_dataset(df)
ic(dataset.head())

# Features and label
X = dataset.iloc[:, :-1] # all the columns except the last
y = dataset.iloc[:, -1]  # last

# Training and testing sets
X1, X2, y1, y2 = train_test_split(
    X, y, test_size=0.2
)

# Learn model
knn = KNeighborsClassifier()
knn.fit(X1, y1)

# Predictions
y_pred = knn.predict(X2)

# Output
print("Test data:\t", " ".join(y2))
print("Prediction:\t", " ".join(y_pred))
print("Score on Train:", knn.score(X1, y1).round(2))
print("Score on Test:", knn.score(X2, y2).round(2))
print("Report:", classification_report(y2, y_pred, zero_division=0))

"""
    ic| 'Initial dataset'
    ic| df.head():                                               Review Sentiment
                0  When you go to the 'Look inside' option, you c...  negative
                1  I wouldn't recommend if completely new to codi...  negative
                2  I bought this book for a master's data science...  positive
                3  Please review the table of contents before pur...  positive
                4  If you are following a parallel course of ML t...  positive
    ic| 'Clear Text (remove numbers, lower, strip)'
    ic| df.head():                                               Review Sentiment
                0  when you go to the 'look inside' option, you c...  negative
                1  i wouldn't recommend if completely new to codi...  negative
                2  i bought this book for a master's data science...  positive
                3  please review the table of contents before pur...  positive
                4  if you are following a parallel course of ml t...  positive
    ic| 'Remove Punctuation'
    ic| df.head():                                               Review Sentiment
                0  when you go to the look inside option you can ...  negative
                1  i wouldnt recommend if completely new to codin...  negative
                2  i bought this book for a masters data science ...  positive
                3  please review the table of contents before pur...  positive
                4  if you are following a parallel course of ml t...  positive
    ic| 'Tokenize Words'
    ic| df['Words'].head(): 0    [when, you, go, to, the, look, inside, option,...
                            1    [i, wouldnt, recommend, if, completely, new, t...
                            2    [i, bought, this, book, for, a, masters, data,...
                            3    [please, review, the, table, of, contents, bef...
                            4    [if, you, are, following, a, parallel, course,...
                            Name: Words, dtype: object
    ic| 'Remove Stop Words'
    ic| df['Words'].head(): 0    [go, look, inside, option, see, code, colors, ...
                            1    [wouldnt, recommend, completely, new, coding, ...
                            2    [bought, book, masters, data, science, class, ...
                            3    [please, review, table, contents, purchasing, ...
                            4    [following, parallel, course, ml, book, great,...
                            Name: Words, dtype: object
    ic| 'Stem Words'
    ic| dataset.head():    abl  abund  access  accompani   account   ...  yet  youll  your  Sentiment
                        0  0.0    0.0     0.0        0.0  0.000000   ...  0.0    0.0   0.0   negative
                        1  0.0    0.0     0.0        0.0  0.220705   ...  0.0    0.0   0.0   negative
                        2  0.0    0.0     0.0        0.0  0.000000   ...  0.0    0.0   0.0   positive
                        3  0.0    0.0     0.0        0.0  0.000000   ...  0.0    0.0   0.0   positive
                        4  0.0    0.0     0.0        0.0  0.000000   ...  0.0    0.0   0.0   positive
                        
                        [5 rows x 566 columns]
    Test data:       negative positive positive negative positive negative negative
    Prediction:      positive positive positive negative positive positive positive
    Score on Train: 0.79
    Score on Test: 0.57
    Report:               precision    recall  f1-score   support

        negative       1.00      0.25      0.40         4
        positive       0.50      1.00      0.67         3

        accuracy                           0.57         7
       macro avg       0.75      0.62      0.53         7
    weighted avg       0.79      0.57      0.51         7
"""