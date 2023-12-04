
"""
The class-based approach might introduce slight overhead due to function calls and attribute access within the class, 
which could lead to slightly slower execution times compared to the procedural version.

The in-place modifications within the procedural version might result in faster execution for smaller datasets.
Each function modifies the data in-place, and the subsequent function operates on the updated data directly.
"""

import pathlib
import pandas as pd
import numpy as np
import sys
import unicodedata
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from icecream import ic
import string

# Dataset
DIR = pathlib.Path(__file__).resolve().parent
df = pd.read_csv(DIR / 'reviews.csv')

class Preprocessing:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.porter = PorterStemmer()

    def preprocess_text(self, text):
        text = text.strip()
        text = re.sub(r"[0-9]", "", text)
        text = text.lower()
        return text

    def remove_punctuation(self, text):
        return text.translate(str.maketrans('', '', string.punctuation))

    def tokenize_words(self, text):
        return word_tokenize(text)

    def remove_stopwords(self, words):
        return [w for w in words if w not in stopwords.words('english')]

    def stem_words(self, words):
        return [self.porter.stem(w) for w in words]

    def preprocess(self, data):
        data['Review'] = data['Review'].apply(self.preprocess_text)
        data['Review'] = data['Review'].apply(self.remove_punctuation)
        data['Words'] = data['Review'].apply(self.tokenize_words)
        data['Words'] = data['Words'].apply(self.remove_stopwords)
        data['Words'] = data['Words'].apply(self.stem_words)
        return data


# Process dataset
procesor = Preprocessing()
df_processed = procesor.preprocess(df)
ic(df_processed.head())

# ------------------------------------------------------------------

# Vectorize dataset
X = procesor.vectorizer.fit_transform([" ".join(words) for words in df_processed['Words']])
y = df_processed['Sentiment']

# Training and testing sets
X1, X2, y1, y2 = train_test_split(
    X, y, test_size=0.2
)

# Learn model
knn = KNeighborsClassifier()
knn.fit(X, y)

# Output score
print("Score on Train:", knn.score(X1, y1).round(2))
print("Score on Test:", knn.score(X2, y2).round(2))

# ------------------------------------------------------------------

# Predictions for unknown review
unknown_review = "Python is becoming the main coding language today."
unknown_processed = procesor.preprocess_text(unknown_review)
unknown_vectorized = procesor.vectorizer.transform([unknown_processed])
ic(unknown_processed)
ic(unknown_vectorized)

predicted_sentiment = knn.predict(unknown_vectorized)
print("Predicted Sentiment:", predicted_sentiment)

# Predictions for unknown review (negative expected)
unknown_review = "Very disappointed with this book. I was expecting more code samples."
unknown_processed = procesor.preprocess_text(unknown_review)
unknown_vectorized = procesor.vectorizer.transform([unknown_processed])

predicted_sentiment = knn.predict(unknown_vectorized)
print("Predicted Sentiment:", predicted_sentiment)
