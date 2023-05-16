""" Decision Tree / Classifier (Play Tennis)

Given certain values for each of the attributes, the learned decision tree 
is able to give a clear answer if weather is suitable or not for tennis.
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
X_new = X.iloc[2:3] # second row in dataset
y_new1 = decision_tree.predict(X_new)[0]

X_new = [2, 2, 0, 0] # third row
X_new = pd.DataFrame([X_new], columns=X.columns)
y_new2 = decision_tree.predict(X_new)[0]

print("Row 2: \n", df.iloc[2:3]); print("Play_prediction:", y_new1, "\n")
print("Row 7: \n", df.iloc[7:8]); print("Play_prediction:", y_new2, "\n")

"""
    Row 2: 
        outlook  temp  humidity  windy  play
    2        0     1         0      0     1
    Play_prediction: 1 

    Row 7: 
        outlook  temp  humidity  windy  play
    7        2     2         0      0     0
    Play_prediction: 0 
"""