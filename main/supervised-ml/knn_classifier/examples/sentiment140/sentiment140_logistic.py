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
df = pd.read_csv(DIR / 'sentiment140.csv', header=None, usecols=[0, 5])

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

# ------------------------------------------------------------------

print("Learning ...")

# Split the dataset into training and testing sets
X = df['Text']
y = df['Sentiment']
X1, X2, y1, y2 = train_test_split(X, y, test_size=0.2, random_state=0)

# Convert text into numerical features
vectorizer = TfidfVectorizer(stop_words='english')
X1 = vectorizer.fit_transform(X1)
X2 = vectorizer.transform(X2)

# Train the model
classifier = LogisticRegression(max_iter=1000)
classifier.fit(X1, y1)

# Make predictions on the test set
y_pred = classifier.predict(X2)

print("Model accuracy:")
print("Sentiment140 / samples =", sample_size)
print("Score on Train:", classifier.score(X1, y1).round(2))
print("Score on Test:", classifier.score(X2, y2).round(2))
print("Report:", classification_report(y2, y_pred), "\n")

# ------------------------------------------------------------------

print("Predict unknown ...")

# Load reviews
df = pd.read_csv(DIR / 'reviews_unknown.csv')
X_unknown = df['Review']
y_unknown = df['Sentiment']

# Convert the sentiment labels
y_unknown = y_unknown.map({'negative': 0, 'positive': 4})

# Convert text into numerical features
X_unknown = vectorizer.transform(X_unknown)

# Make predictions on unknown set
y_unknown_pred = classifier.predict(X_unknown)
score_unknown = classifier.score(X_unknown, y_unknown)

print("Reviews (unknown):")
print("Unknown:\t", y_unknown.values)
print("Prediction:\t", y_unknown_pred)
print("Score on Unknown:", score_unknown, round(2))
print("Report:", classification_report(y_unknown, y_unknown_pred), "\n")

# ------------------------------------------------------------------

print("Predict review:")

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
y_unknown_pred = classifier.predict(X_unknown)
print("Prediction:", labels[y_unknown_pred[0]])

"""
	Loading ...
	Preprocessing ...
	Learning ...
	Predict unknown ...
    
	Model accuracy:
	Sentiment140 / samples = 100000
	Score on Train: 0.84
	Score on Test: 0.76
	Report:               precision    recall  f1-score   support

		       0       0.77      0.74      0.75     10075
		       4       0.75      0.77      0.76      9925

		accuracy                           0.76     20000
	   macro avg       0.76      0.76      0.76     20000
	weighted avg       0.76      0.76      0.76     20000
	 
	Reviews (unknown):
	Unknown:         [0 0 4 4 4 4 4 4 4 4 4 4 4 0 0 0 4 4 4 4 4 4 0 0 0 0 0 0 0 0 0 0 4 4 0]
	Prediction:      [0 0 4 4 4 0 4 4 4 0 4 4 4 4 0 4 4 4 4 4 4 4 4 4 0 0 0 4 4 0 0 4 4 4 4]
	Score on Unknown: 0.71
	Report:               precision    recall  f1-score   support

		       0       0.80      0.50      0.62        16
		       4       0.68      0.89      0.77        19

		accuracy                           0.71        35
	   macro avg       0.74      0.70      0.69        35
	weighted avg       0.73      0.71      0.70        35
	 
	Predict review ...
	Review:
	 Utter waste of money. As someone that has done a fair bit of coding in the past I wanted 
     an introductory text to Python - something that would outline what could really be done with 
     the programming language and provide an entry point to it. What a waste. 
     The book is elementary on the one hand - and yet manages to miss a ton of required detail to 
     actually do anything useful. Don't waste your time with it.

	Expected: negative
	Prediction: negative
"""
