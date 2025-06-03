""" Tic Tac Toe / Decistion Tree

Values: x = first player, o = second player, b = blank
Target: 1 means that X wins

If you provide a current game state to the Decision Tree model, 
it will predict the outcome or score for the end state, 
assuming optimal play from both players. 

However, it does not directly provide the best move to make 
in the current state.
"""

import pathlib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

DIR = pathlib.Path(__file__).resolve().parent
df = pd.read_csv(DIR / "data/tic-tac-toe-endgame-uci.csv")

# Encode lables
LE = LabelEncoder()
df_encoded = pd.DataFrame()
for col in df.columns:
    df_encoded[col] = LE.fit_transform(df[col])

# Train data
X = df_encoded.iloc[:,0:-1]
Y = df_encoded['V10']

# Spliting data (train and test)
X1, X2, Y1, Y2 = train_test_split(X, Y, test_size=0.3, random_state=42)

# Fitting the model
model = DecisionTreeClassifier(random_state=42)
model.fit(X1, Y1)
score = model.score(X2, model.predict(X2)) # on test dataset

# Predictions
board = ["X", " ", "X",
         "X", "O", " ",
         "O", " ", " "] # O to move / expected 0 (Draw)
x_new = [2, 0, 2,
         2, 1, 0,
         1, 0, 0] 
x_new = pd.DataFrame([x_new], columns=X.columns)
y_pred = model.predict(x_new) # 0

# Output
output = [
    ["Dataset:", df],
    ["Encoded:", df_encoded],
    ["Features:", X],
    ["Targets:", Y],
    ["Classes (X win) yes/no:", Y.unique()],
    ["Accuracy score:", score],
    ["Unknown x_new:", x_new],
    ["Prediction class:", y_pred],
]
for v in output:
    print("\n", v[0]); print(v[1])

"""
    Dataset:
        V1 V2 V3 V4 V5 V6 V7 V8 V9       V10
    0    x  x  x  x  o  o  x  o  o  positive
    1    x  x  x  x  o  o  o  x  o  positive
    2    x  x  x  x  o  o  o  o  x  positive
    3    x  x  x  x  o  o  o  b  b  positive
    4    x  x  x  x  o  o  b  o  b  positive
    ..  .. .. .. .. .. .. .. .. ..       ...
    953  o  x  x  x  o  o  o  x  x  negative
    954  o  x  o  x  x  o  x  o  x  negative
    955  o  x  o  x  o  x  x  o  x  negative
    956  o  x  o  o  x  x  x  o  x  negative
    957  o  o  x  x  x  o  o  x  x  negative

    [958 rows x 10 columns]

    Encoded:
        V1  V2  V3  V4  V5  V6  V7  V8  V9  V10
    0     2   2   2   2   1   1   2   1   1    1
    1     2   2   2   2   1   1   1   2   1    1
    2     2   2   2   2   1   1   1   1   2    1
    3     2   2   2   2   1   1   1   0   0    1
    4     2   2   2   2   1   1   0   1   0    1
    ..   ..  ..  ..  ..  ..  ..  ..  ..  ..  ...
    953   1   2   2   2   1   1   1   2   2    0
    954   1   2   1   2   2   1   2   1   2    0
    955   1   2   1   2   1   2   2   1   2    0
    956   1   2   1   1   2   2   2   1   2    0
    957   1   1   2   2   2   1   1   2   2    0

    [958 rows x 10 columns]

    Features:
        V1  V2  V3  V4  V5  V6  V7  V8  V9
    0     2   2   2   2   1   1   2   1   1
    1     2   2   2   2   1   1   1   2   1
    2     2   2   2   2   1   1   1   1   2
    3     2   2   2   2   1   1   1   0   0
    4     2   2   2   2   1   1   0   1   0
    ..   ..  ..  ..  ..  ..  ..  ..  ..  ..
    953   1   2   2   2   1   1   1   2   2
    954   1   2   1   2   2   1   2   1   2
    955   1   2   1   2   1   2   2   1   2
    956   1   2   1   1   2   2   2   1   2
    957   1   1   2   2   2   1   1   2   2

    [958 rows x 9 columns]

    Targets:
    0      1
    1      1
    2      1
    3      1
    4      1
        ..
    953    0
    954    0
    955    0
    956    0
    957    0
    Name: V10, Length: 958, dtype: int64

    Classes (X win) yes/no:
    [1 0]

    Accuracy score:
    1.0

    Unknown x_new:
    V1  V2  V3  V4  V5  V6  V7  V8  V9
    0   2   0   1   2   1   0   1   0   0

    Prediction class:
    [0]
"""