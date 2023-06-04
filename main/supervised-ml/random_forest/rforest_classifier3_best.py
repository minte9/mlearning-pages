""" Random Forest / Classifier (Play Tenis) - Best

Random Forests try to fix this overfitting by using multiple decision trees 
that are slightly different and averaging the results.
With GridSearchCV we can see what params fit better.
"""

import pathlib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

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
X1, X2, Y1, Y2 = train_test_split(X, Y, test_size=0.2, random_state=42)

# --------------------------------------------------------------------

# Best parameters
parameters = {
    'max_depth': [2, 3, 4, 10],
    'n_estimators': [5, 10, 20]
}
model = RandomForestClassifier(random_state=42)
grid = GridSearchCV(model, parameters, cv=3)
grid.fit(X1, Y1)

print("Best Parameters:", grid.best_params_) # max_depth: 2, n_estimators: 10
print("Best Score:", round(grid.best_score_, 2), "\n")

# --------------------------------------------------------------------

# Fitting best model
forest_model = RandomForestClassifier(n_estimators=10, max_depth=2, random_state=0)
forest_model.fit(X1, Y1)
forest_score = forest_model.score(X2, Y2)

# Prediction
x_new =  X2.iloc[2] # expect 0
x_new = pd.DataFrame([x_new], columns=X2.columns)
y_pred = forest_model.predict(x_new)[0]
assert y_pred == 0

print("Test Data:"); print(X2, "\n")
print("Encoded:"); print(Y2, "\n")
print("Unknown:"); print(x_new, "\n")
print("Prediction:", y_pred)
print("Score:", round(forest_score,2))

"""
    Best Parameters: {'max_depth': 2, 'n_estimators': 10}
    Best Score: 0.72

    Test Data:
        outlook  temp  humidity  windy
    9         1     2         1      0
    11        0     2         0      1
    0         2     1         0      0 

    Encoded:
    9     1
    11    1
    0     0
     Name: play

    Unknown:
        outlook  temp  humidity  windy
    0        2     1         0      0 

    Prediction: 0
    Score: 1.0
"""