import pathlib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import random

print("Loading ...")

# Load the dataset
DIR = pathlib.Path(__file__).resolve().parent
df = pd.read_csv(DIR / 'sentiment140.csv', header=None, usecols=[0, 5])

# Assign column names
df.columns = ['Sentiment', 'Text']
print(df.head())

# ----------------------------------------------------------

print("Learning ...")

# Sample
sample_size = 10_000
sample_indices = random.sample(range(len(df)), sample_size)
df = df.iloc[sample_indices]

# Split the dataset into training and testing sets
X = df['Text']
y = df['Sentiment']
X1, X2, y1, y2 = train_test_split(X, y, test_size=0.2, random_state=0)

# Convert text into numerical features
vectorizer = TfidfVectorizer() # CountVectorizer()
X1 = vectorizer.fit_transform(X1)
X2 = vectorizer.transform(X2)

# Train the KNN classifier
knn = KNeighborsClassifier()
knn.fit(X1, y1)

# Make predictions on the test set
y_pred = knn.predict(X2)

# ----------------------------------------------------------

# Load reviews
df = pd.read_csv(DIR / 'reviews_unknown.csv')
X_unknown = df['Review']
y_unknown = df['Sentiment']

# Convert the sentiment labels
y_unknown = y_unknown.map({'negative': 0, 'positive': 4})

# Convert text into numerical features
X_unknown = vectorizer.transform(X_unknown)

# Make predictions on unknown set
y_unknown_pred = knn.predict(X_unknown)

# ----------------------------------------------------------

print("Sentiment140 / samples =", sample_size)
print("Score on Train:", knn.score(X1, y1).round(2))
print("Score on Test:", knn.score(X2, y2).round(2))
print("Report:", classification_report(y2, y_pred), "\n")

print("Reviews (unknown):")
print("Unknown:\t", y_unknown.values)
print("Prediction:\t", y_unknown_pred)
print("Score on Unknown:", knn.score(X_unknown, y_unknown).round(2))
print("Report:", classification_report(y_unknown, y_unknown_pred), "\n")

# ------------------------------------------------------------------

# Load one row from reviews
df = pd.read_csv(DIR / 'reviews_unknown.csv')
random_index = df.sample().index.item()

df['Sentiment'] = df['Sentiment'].map({'negative': 0, 'positive': 4})

X_unknown = [df['Review'][random_index]]
y_unknown = df['Sentiment'][random_index]

labels = {0: 'negative', 4: 'positive'}

print("Review:\n", X_unknown[0])
print("Expected:", labels[y_unknown])

# Convert using vectorizer
X_unknown = vectorizer.transform(X_unknown)
y_unknown_pred = knn.predict(X_unknown)

print("Prediction:", labels[y_unknown_pred[0]])

"""
	Loading ...
	Learning ...
	Sentiment140 / samples = 10000
	Score on Train: 0.78
	Score on Test: 0.65
	Report:               precision    recall  f1-score   support

		       0       0.65      0.71      0.68      1023
		       4       0.66      0.59      0.63       977

		accuracy                           0.65      2000
	   macro avg       0.65      0.65      0.65      2000
	weighted avg       0.65      0.65      0.65      2000
	 

	Reviews (unknown):
	Unknown:         [0 0 4 4 4 4 4 4 4 4 4 4 4 0 0 0 4 4 4 4 4 4 0 0 0 0 0 0 0 0 0 0 4 4 0]
	Prediction:      [0 0 0 0 0 0 4 0 0 0 4 0 4 4 0 0 4 0 0 0 4 4 0 0 0 0 0 4 0 0 0 0 4 4 0]
	Score on Unknown: 0.63
	Report:               precision    recall  f1-score   support

		       0       0.56      0.88      0.68        16
		       4       0.80      0.42      0.55        19

		accuracy                           0.63        35
	   macro avg       0.68      0.65      0.62        35
	weighted avg       0.69      0.63      0.61        35
	 

	Review:
	 very disappointed with this book. i was expecting more code samples but the author was too busy with other less trivial things.
	Expected: negative
	Prediction: negative
"""
