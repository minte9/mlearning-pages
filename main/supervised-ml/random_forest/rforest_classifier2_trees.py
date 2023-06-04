""" Random Forest / Classifier (Play Tenis) - Trees

Random Forests try to fix this overfitting by using multiple decision trees 
that are slightly different and averaging the results.

With Cross-validation we put aside 10% of the data, before train.
Those test data will be use to measure how good the model is.
"""

import pathlib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree

# Dataset
DIR = pathlib.Path(__file__).resolve().parent
df = pd.read_csv(DIR / 'data/play_tennis.csv')

# Encode lables
df_encoded = pd.DataFrame()
for col in df.columns:
    df_encoded[col] = LabelEncoder().fit_transform(df[col])

# Train and test data
X = df_encoded.drop(columns=["play"])
Y = df_encoded['play']
X1, X2, Y1, Y2 = train_test_split(X, Y, random_state=42)

# Fitting the model
forest_model = RandomForestClassifier(max_depth=3, random_state=66)
forest_model.fit(X1, Y1)
forest_score = forest_model.score(X2, Y2)

# Prediction
x_new =  X2.iloc[0] # expect 1
x_new = pd.DataFrame([x_new], columns=X2.columns)
y_pred = forest_model.predict(x_new)[0]
assert y_pred == 1

# Output
output_trees = []
for i, t in enumerate(forest_model.estimators_):
    out_tree = tree.export_text(t, feature_names=list(X.columns))
    output_trees.append(out_tree)

print("Test Data:"); print(X2, "\n")
print("Encoded:"); print(Y2, "\n")
print("DecisionTree 1"); print(output_trees[0])
print("DecisionTree 2"); print(output_trees[1])
print("DecisionTree 3"); print(output_trees[2])
print("Unknown:"); print(x_new, "\n")
print("Prediction:", y_pred)
print("Score:", round(forest_score,2))

"""
	Test Data:
		outlook  temp  humidity  windy
	9         1     2         1      0
	11        0     2         0      1
	0         2     1         0      0
	12        0     1         1      0 

	Encoded:
	9     1
	11    1
	0     0
	12    1
	Name: play, dtype: int64 

	DecisionTree 1
	|--- outlook <= 0.50
	|   |--- class: 1.0
	|--- outlook >  0.50
	|   |--- temp <= 1.00
	|   |   |--- windy <= 0.50
	|   |   |   |--- class: 1.0
	|   |   |--- windy >  0.50
	|   |   |   |--- class: 0.0
	|   |--- temp >  1.00
	|   |   |--- windy <= 0.50
	|   |   |   |--- class: 1.0
	|   |   |--- windy >  0.50
	|   |   |   |--- class: 0.0

	DecisionTree 2
	|--- windy <= 0.50
	|   |--- outlook <= 1.50
	|   |   |--- class: 1.0
	|   |--- outlook >  1.50
	|   |   |--- class: 0.0
	|--- windy >  0.50
	|   |--- temp <= 1.50
	|   |   |--- class: 0.0
	|   |--- temp >  1.50
	|   |   |--- class: 1.0

	DecisionTree 3
	|--- humidity <= 0.50
	|   |--- temp <= 1.50
	|   |   |--- class: 1.0
	|   |--- temp >  1.50
	|   |   |--- windy <= 0.50
	|   |   |   |--- class: 0.0
	|   |   |--- windy >  0.50
	|   |   |   |--- class: 0.0
	|--- humidity >  0.50
	|   |--- class: 1.0

	Unknown:
	   outlook  temp  humidity  windy
	9        1     2         1      0 

	Prediction: 1
	Score: 0.75
"""