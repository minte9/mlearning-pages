""" Decision Tree / Classifier (Play Tennis)

When we don't have any data to test, we can just let the model 
to predict our train data.
"""

import numpy as np
import pandas as pd
import pathlib
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
import graphviz

# Dataset
DIR = pathlib.Path(__file__).resolve().parent
df = pd.read_csv(DIR / 'data/play_tennis.csv')

# Encode data
for col in df.columns:
    df[col] = LabelEncoder().fit_transform(df[col])

# Train data
X = df.drop(['play'], axis=1)
y = df['play']

# Fitting the model
decision_tree = DecisionTreeClassifier(criterion='entropy')
decision_tree.fit(X, y)

# Predictions
y_pred = decision_tree.predict(X)
score = decision_tree.score(X, y)

print("Train data:", y.values)
print("Predictions:", y_pred)
print("Score accuracy:", score)

"""
    Train data: [0 0 1 1 1 0 1 0 1 1 1 1 1 0]
    Predictions: [0 0 1 1 1 0 1 0 1 1 1 1 1 0]
    Score accuracy: 1.0
"""