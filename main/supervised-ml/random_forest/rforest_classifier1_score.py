""" Random Forest / Classifier (Tic Tac Toe)

Random Forests are using subsets of the dataset, randomly selected.
The generated trees have unique set of data.
The data is selected randomly for the original dataset, with replacement.
Each one of the tree has the same size as the original data.

With Cross-validation we put aside 25% of the data, before train.
Those test data will be use to measure how good the model is.
"""

import pathlib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Dataset
DIR = pathlib.Path(__file__).resolve().parent
df = pd.read_csv(DIR / 'data/ttt_dataset.csv')

# Encode lables
df_encoded = pd.DataFrame()
for col in df.columns:
    df_encoded[col] = LabelEncoder().fit_transform(df[col])

# Train and test data
X = df_encoded.drop(columns=["score", "is_terminal"])
Y = df_encoded['score']
X1, X2, Y1, Y2 = train_test_split(X, Y, test_size=0.25, random_state=0)

# ---------------------------------------------------

# Fitting the model
forest_model = RandomForestClassifier(n_estimators=30)
forest_model.fit(X1, Y1)
forest_score = forest_model.score(X2, Y2)

# ---------------------------------------------------

# Prediction
x_new =  X2.iloc[0]
x_new = pd.DataFrame([x_new], columns=X2.columns)
y_pred = forest_model.predict(x_new)[0]
assert y_pred == 2

print("Test Data:"); print(X2, "\n")
print("Encoded:"); print(Y2, "\n")
print("Unknown:"); print(x_new, "\n")
print("Prediction:", y_pred)
print("Score:", round(forest_score,2))

"""
    Test Data:
            V1  V2  V3  V4  V5  V6  V7  V8  V9
    4534    2   2   2   0   0   1   1   0   0
    3544    1   2   0   1   0   0   2   0   2
    5287    1   2   2   2   0   2   1   1   1
    427     2   1   2   2   2   1   0   1   1
    5969    2   1   2   2   2   0   1   1   1
    ...    ..  ..  ..  ..  ..  ..  ..  ..  ..
    28369   1   2   1   0   1   2   0   2   2
    2253    1   2   2   2   0   0   0   1   0
    2323    1   2   2   2   1   1   2   1   2
    21741   2   1   0   1   0   0   2   2   1
    10353   2   0   1   2   2   2   1   1   0

    [7265 rows x 9 columns] 

    Encoded:
    4534     2
    3544     2
    5287     0
    427      2
    5969     0
            ..
    28369    0
    2253     0
    2323     1
    21741    1
    10353    2
    Name: score

    Unknown:
          V1  V2  V3  V4  V5  V6  V7  V8  V9
    4534   2   2   2   0   0   1   1   0   0 

    Prediction: 2
    Score: 0.97
"""