import pathlib
import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

print("Loading ...")

# Load the dataset
DIR = pathlib.Path(__file__).resolve().parent
df = pd.read_csv(DIR / 'data/sentiment140.csv', header=None, usecols=[0, 5])

# Assign column names
df.columns = ['Sentiment', 'Text']

# Sample
sample_size = 100_000
sample_indices = random.sample(range(len(df)), sample_size)
df = df.iloc[sample_indices]

# ------------------------------------------------------------------

print("Preprocessing ...")

# Preprocessing operations
df['Text'] = df['Text'].str.lower()  # Convert text to lowercase
df['Text'] = df['Text'].str.replace('[^\w\s]', '', regex=True)  # Remove punctuation
df['Text'] = df['Text'].str.replace('\d+', '', regex=True)  # Remove numbers
df['Text'] = df['Text'].str.strip()  # Strip whitespaces

print("Learning ...")

# Split the dataset into training and testing sets
X = df['Text']
y = df['Sentiment']
X1, X2, y1, y2 = train_test_split(X, y, test_size=0.2, random_state=0)

# Convert text into numerical features
vectorizer = TfidfVectorizer(stop_words='english')
X1 = vectorizer.fit_transform(X1)
X2 = vectorizer.transform(X2)

# Train the KNN classifier
knn = LogisticRegression(max_iter=1000)
knn.fit(X1, y1)

# Make predictions on the test set
y_pred = knn.predict(X2)

# ------------------------------------------------------------------

print("Predict unknown ...")

# Load reviews
df = pd.read_csv(DIR / 'data/reviews.csv')
X_unknown = df['Review']
y_unknown = df['Sentiment']

# Convert the sentiment labels
y_unknown = y_unknown.map({'negative': 0, 'positive': 4})

# Convert text into numerical features
X_unknown = vectorizer.transform(X_unknown)

# Make predictions on unknown set
y_unknown_pred = knn.predict(X_unknown)

# ------------------------------------------------------------------

print("Model accuracy ...")

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

print("Predict review ...")

# Load one row from reviews
df = pd.read_csv(DIR / 'data/reviews.csv')
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
