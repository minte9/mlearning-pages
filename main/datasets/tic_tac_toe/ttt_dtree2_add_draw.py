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
df = pd.read_csv(DIR / "data/tic-tac-toe-somesh24.csv")

def add_draw_class():
    for i in range(len(df)):
        win = False
        for j in range(3):
            if df.iloc[i][j] == df.iloc[i][j+1] == df.iloc[i][j+2] and df.iloc[i][j+1] != 'b':
                win = True # horizontal win
            if df.iloc[i][j] == df.iloc[i][j+3] == df.iloc[i][j+6] and df.iloc[i][j+1] != 'b':
                win = True # vertical win

        if df.iloc[i][0] == df.iloc[i][4] == df.iloc[i][9] or \
            df.iloc[i][2] == df.iloc[i][4] == df.iloc[i][6] and df.iloc[i][4] != 'b':
                win = True  # diagonal win
        if not win:
            df.loc[i, 'class'] = 'Draw'
    return df

add_draw_class()

# Convert boolean values to strings in the 'class' column
df['class'] = df['class'].astype(str)

# Encode lables
LE = LabelEncoder()
df_encoded = pd.DataFrame()
for col in df.columns:
    df_encoded[col] = LE.fit_transform(df[col])

# Train data
X = df_encoded.iloc[:,0:-1]
Y = df_encoded['class'] # [0, 1, 2] / draw, O win, X win

# Spliting data (train and test)
X1, X2, Y1, Y2 = train_test_split(X, Y, test_size=0.3, random_state=42)

# Fitting the model
model = DecisionTreeClassifier(random_state=42)
model.fit(X1, Y1)
score = model.score(X2, model.predict(X2))

# Predictions
board = ([
    "X", " ", "X",
    "X", "O", " ",
    "O", " ", " "], False, 0) # O to move / expected 0 (Draw)
x_new = [2,0,2,2,1,0,1,0,0] # 0
x_new = pd.DataFrame([x_new], columns=X.columns)
y_pred = model.predict(x_new)
assert y_pred == 0

# Output
output = [
    ["Dataset:", df],
    ["Encoded:", df_encoded],
    ["Features:", X],
    ["Targets:", Y],
    ["Classes (X win, Draw, O win):", Y.unique()],
    ["Accuracy score:", score],
    ["Unknown x_new:", x_new],
    ["Prediction class:", y_pred],
]
for v in output:
    print("\n", v[0]); print(v[1])

"""
    Dataset:
        TL TM TR ML MM MR BL BM BR  class
    0    x  x  x  x  o  o  x  o  o   True
    1    x  x  x  x  o  o  o  x  o   True
    2    x  x  x  x  o  o  o  o  x   True
    3    x  x  x  x  o  o  o  b  b   True
    4    x  x  x  x  o  o  b  o  b   True
    ..  .. .. .. .. .. .. .. .. ..    ...
    953  o  x  x  x  o  o  o  x  x  False
    954  o  x  o  x  x  o  x  o  x   Draw
    955  o  x  o  x  o  x  x  o  x   Draw
    956  o  x  o  o  x  x  x  o  x   Draw
    957  o  o  x  x  x  o  o  x  x  False

    [958 rows x 10 columns]

    Encoded:
        TL  TM  TR  ML  MM  MR  BL  BM  BR  class
    0     2   2   2   2   1   1   2   1   1      2
    1     2   2   2   2   1   1   1   2   1      2
    2     2   2   2   2   1   1   1   1   2      2
    3     2   2   2   2   1   1   1   0   0      2
    4     2   2   2   2   1   1   0   1   0      2
    ..   ..  ..  ..  ..  ..  ..  ..  ..  ..    ...
    953   1   2   2   2   1   1   1   2   2      1
    954   1   2   1   2   2   1   2   1   2      0
    955   1   2   1   2   1   2   2   1   2      0
    956   1   2   1   1   2   2   2   1   2      0
    957   1   1   2   2   2   1   1   2   2      1

    [958 rows x 10 columns]

    Features:
        TL  TM  TR  ML  MM  MR  BL  BM  BR
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
    0      2
    1      2
    2      2
    3      2
    4      2
        ..
    953    1
    954    0
    955    0
    956    0
    957    1
    Name: class, Length: 958, dtype: int64

    Classes (X win, Draw, O win):
    [2 0 1]

    Accuracy score:
    1.0

    Unknown x_new:
    TL  TM  TR  ML  MM  MR  BL  BM  BR
    0   2   0   2   2   1   0   1   0   0

    Prediction class:
    [0]
"""