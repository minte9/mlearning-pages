""" Decision Tree / Classifier (Play Tennis)

Given certain values for each of the attributes, the learned decision tree 
is able to give a clear answer if weather is suitable or not for tennis.
"""

import pathlib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# Dataset
DIR = pathlib.Path(__file__).resolve().parent
df = pd.read_csv(DIR / 'data/play_tennis.csv')

# Encode lables
for col in df.columns:
    df[col] = LabelEncoder().fit_transform(df[col])

# Train data
X = df.drop(columns=["play"])
Y = df['play']

# Fitting the model
dtree_model = DecisionTreeClassifier(random_state=42)
dtree_model.fit(X, Y)

# Prediction
x_new =  [1, 0, 1, 0] # expect 1
x_new = pd.DataFrame([x_new], columns=X.columns)
y_pred = dtree_model.predict(x_new)[0]

print("Prediction for:"); print(x_new.to_markdown())
print("Class:", y_pred)


"""
    Prediction for:
    |    |   outlook |   temp |   humidity |   windy |
    |---:|----------:|-------:|-----------:|--------:|
    |  0 |         1 |      0 |          1 |       0 |
    Class: 1
"""