import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from icecream import ic
import string
from joblib import dump, load

# Processor class blueprint
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

import pathlib
DIR = pathlib.Path(__file__).resolve().parent

# Load trained model to make predictions
knn = load(DIR / 'reviews_knn_model.joblib')

# Load the procesor instance
procesor = load(DIR / 'reviews_knn_procesor.joblib')
ic(procesor.vectorizer)

# Prediction
unknown_review = "Very disappointed with this book. I was expecting more code samples."
unknown_processed = procesor.preprocess_text(unknown_review)
unknown_vectorized = procesor.vectorizer.transform([unknown_processed])

predicted_sentiment = knn.predict(unknown_vectorized)
print("Predicted Sentiment:", predicted_sentiment)

"""
    ic| procesor.vectorizer: TfidfVectorizer()
    Predicted Sentiment: ['negative']
"""