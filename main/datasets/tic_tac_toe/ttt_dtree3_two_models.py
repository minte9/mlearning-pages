""" Tic Tac Toe / Decistion Tree

Even though applying machine learning to a problem which can be easily solved 
by deterministic methods doesn’t add much value, 
it can still be considered a good problem for learning ML.
"""

import pathlib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

DIR = pathlib.Path(__file__).resolve().parent
df = pd.read_csv(DIR / "data/tic-tac-toe-amit9oc3.csv")

# Train data
X1 = df[(df.IsO == 1)].drop(columns=["IsO", "score"]) # O last move
Y1 = df[(df.IsO == 1)]['score']
X2 = df[(df.IsO == 0)].drop(columns=["IsO", "score"]) # X last move
Y2 = df[(df.IsO == 0)]['score']

# Fitting the models (X and O)
dtreeX = DecisionTreeClassifier()
dtreeX.fit(X1, Y1)
dtreeO = DecisionTreeClassifier()
dtreeO.fit(X2, Y2)

# Predictions
board = ["X", "O", "X", 
         "O", "X", " ",
         "O", "X", " "] # O to move / expected 0 (Draw)
x_new = [1,0,1,0,1,0,0,1,0,
         0,1,0,1,0,0,1,0,0]
x_new = pd.DataFrame([x_new], columns=X1.columns)
y_pred = dtreeO.predict(x_new) # 0
assert y_pred == 0

board = ["X", " ", " ",
         "X", "O", " ",
         "O", " ", " "] # X to move / expected 0 (Draw)
x_new = [1, 0, 0, 1, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 1, 0, 1, 0, 0]
x_new = pd.DataFrame([x_new], columns=X1.columns)
y_pred = dtreeX.predict(x_new) # 0
assert y_pred == 0

# Output
output = [
    ["Dataset:", df],
    ["Features:", X1],
    ["Targets:", Y1],
    ["Unique Classes (draw, O_win, X_win):", Y1.unique()],
    ["Unknown x_new:", x_new],
    ["Prediction class:", y_pred],
]
for v in output:
    print("\n", v[0]); print(v[1])

"""
    Dataset:
        X0  X1  X2  X3  X4  X5  X6  X7  X8  O0  O1  O2  O3  O4  O5  O6  O7  O8  IsO  score
    0       0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   1    1      0
    1       0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   1   0    1      0
    2       0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   1   0   0    1      0
    3       0   0   0   0   0   0   0   0   0   0   0   0   0   0   1   0   0   0    1      0
    4       0   0   0   0   0   0   0   0   0   0   0   0   0   1   0   0   0   0    1      0
    ...    ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ...    ...
    10949   1   1   1   1   0   0   0   0   0   0   0   0   0   1   1   1   0   1    0    100
    10950   1   1   1   1   0   0   0   0   0   0   0   0   0   1   1   1   1   0    0    100
    10951   1   1   1   1   0   0   0   0   1   0   0   0   0   1   1   1   1   0    0    100
    10952   1   1   1   1   0   0   0   1   0   0   0   0   0   1   1   1   0   1    0    100
    10953   1   1   1   1   0   0   1   0   0   0   0   0   0   1   1   0   1   1    0    100

    [10954 rows x 20 columns]

    Features:
        X0  X1  X2  X3  X4  X5  X6  X7  X8  O0  O1  O2  O3  O4  O5  O6  O7  O8
    0       0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   1
    1       0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   1   0
    2       0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   1   0   0
    3       0   0   0   0   0   0   0   0   0   0   0   0   0   0   1   0   0   0
    4       0   0   0   0   0   0   0   0   0   0   0   0   0   1   0   0   0   0
    ...    ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..
    10831   1   1   0   1   1   0   0   0   0   0   0   1   0   0   0   1   1   1
    10833   1   1   0   1   1   0   0   0   0   0   0   1   0   0   1   0   1   1
    10835   1   1   0   1   1   0   0   0   0   0   0   1   0   0   1   1   0   1
    10837   1   1   0   1   1   0   0   0   0   0   0   1   0   0   1   1   1   0
    10838   1   1   0   1   1   0   0   0   0   0   0   1   0   0   1   1   1   1

    [5477 rows x 18 columns]

    Targets:
    0          0
    1          0
    2          0
    3          0
    4          0
            ... 
    10831    100
    10833    100
    10835    100
    10837   -100
    10838    100
    Name: score, Length: 5477, dtype: int64

    Unique Classes (draw, O_win, X_win):
    [   0 -100  100]

    Unknown x_new:
    X0  X1  X2  X3  X4  X5  X6  X7  X8  O0  O1  O2  O3  O4  O5  O6  O7  O8
    0   1   0   0   1   0   0   0   0   0   0   0   0   0   1   0   1   0   0

    Prediction class:
    [0]
"""