""" Reviews Sentiment (KNN)
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

def debug(s1, s2=None, debug=False):
    if debug:
        print(s1, s2)

debug("Initial Dataset:\n", df)
df['Review'] = clear_text(df['Review']);            debug("Cleanning:\n", df)
df['Review'] = remove_punctuation(df['Review']);    debug("Remove Punctuation:\n", df)
df['Words']  = tokenize_words(df['Review']);        debug("Tokenize Words:\n", df['Words'])
df['Words']  = remove_stopwords(df['Words']);       debug("Remove Stop Words:\n", df['Words'])
df['Words']  = steming_words(df['Words']);          debug("Stem Words:\n", df['Words'])

def build_dataset(df):
    # E = CountVectorizer() # Bag of words
    E = TfidfVectorizer() # Word importance

    # Features matrix
    F = E.fit_transform([" ".join(words) for words in df['Words']])
    T = pd.DataFrame(F.toarray(), columns=E.get_feature_names_out())

    # Dataset
    dataset = pd.concat([T, df['Sentiment']], axis=1)
    return dataset

# Final dataset
dataset = build_dataset(df)
debug("Final Dataset:\n\n", dataset)

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
	Test data:       positive negative positive negative positive negative positive
	Prediction:      positive positive positive positive positive positive positive
	Score on Train: 0.75
	Score on Test: 0.57
	Report:               precision    recall  f1-score   support

		negative       0.00      0.00      0.00         3
		positive       0.57      1.00      0.73         4

		accuracy                           0.57         7
	   macro avg       0.29      0.50      0.36         7
	weighted avg       0.33      0.57      0.42         7
"""