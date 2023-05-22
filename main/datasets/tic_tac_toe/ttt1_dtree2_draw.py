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
df = pd.read_csv(DIR / "data/tic-tac-toe-endgame.csv")

def add_draw_class(df):
    for i in range(len(df)):
        win = False

        for j in range(3):
            if df.iloc[i][j] == df.iloc[i][j+1] == df.iloc[i][j+2] and df.iloc[i][j+1] != 'b':
                win = True # horizontal win

        for j in range(3):
            if df.iloc[i][j] == df.iloc[i][j+3] == df.iloc[i][j+6] and df.iloc[i][j+1] != 'b':
                win = True # vertical win
        
        if df.iloc[i][0] == df.iloc[i][4] == df.iloc[i][9] or \
            df.iloc[i][2] == df.iloc[i][4] == df.iloc[i][6] and df.iloc[i][4] != 'b':
                win = True  # diagonal win

        if not win:
            df.loc[i]['V10'] = 'draw'
    return df

df = add_draw_class(df)

# Encode lables
LE = LabelEncoder()
df_encoded = pd.DataFrame()
for col in df.columns:
    df_encoded[col] = LE.fit_transform(df[col])

# Train data
X = df_encoded.iloc[:,0:-1]
Y = df_encoded['V10'] # [0, 1, 2] / draw, O win, X win

add_draw_class(df_encoded)

# Spliting data (train and test)
X1, X2, Y1, Y2 = train_test_split(X, Y, test_size=0.3, random_state=42)

# Fitting the model
model = DecisionTreeClassifier(random_state=42)
model.fit(X1, Y1)

# Prediction
new_board, player, expected = ([
    "X", "O", " ",
    "X", "X", "O",
    "O", " ", " "], True, 1) # X to move / expected 2 (X wins)
x_new = [2,1,0,2,2,1,1,0,0] # 2

new_board, player, expected = ([
    "X", " ", "X",
    "X", "O", " ",
    "O", " ", " "], True, 1) # O to move / expected 0 (Draw)
x_new = [2,0,2,2,1,0,1,0,0] # 0

x_new = pd.DataFrame([x_new], columns=X.columns)
y_pred = model.predict(x_new)

# Accuracy score
score = model.score(X2, model.predict(X2))

# Output
output = [
    ["Dataset:", df.head(4).to_markdown()],
    ["Encoded:", df_encoded.head(4).to_markdown()],
    ["Features:", X.head(4).to_markdown()],
    ["Targets:", Y.head(4).to_markdown()],
    ["Unique targets:", Y.unique()],
    ["Unknown x_new:", x_new],
    ["Prediction class:", y_pred],
    ["Accuracy score:", score],
]
for v in output:
    print("\n", v[0]); print(v[1])

"""
    Dataset:
    |    | V1   | V2   | V3   | V4   | V5   | V6   | V7   | V8   | V9   | V10      |
    |---:|:-----|:-----|:-----|:-----|:-----|:-----|:-----|:-----|:-----|:---------|
    |  0 | x    | x    | x    | x    | o    | o    | x    | o    | o    | positive |
    |  1 | x    | x    | x    | x    | o    | o    | o    | x    | o    | positive |
    |  2 | x    | x    | x    | x    | o    | o    | o    | o    | x    | positive |
    |  3 | x    | x    | x    | x    | o    | o    | o    | b    | b    | positive |

    Encoded:
    |    |   V1 |   V2 |   V3 |   V4 |   V5 |   V6 |   V7 |   V8 |   V9 |   V10 |
    |---:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|------:|
    |  0 |    2 |    2 |    2 |    2 |    1 |    1 |    2 |    1 |    1 |     1 |
    |  1 |    2 |    2 |    2 |    2 |    1 |    1 |    1 |    2 |    1 |     1 |
    |  2 |    2 |    2 |    2 |    2 |    1 |    1 |    1 |    1 |    2 |     1 |
    |  3 |    2 |    2 |    2 |    2 |    1 |    1 |    1 |    0 |    0 |     1 |

    Features:
    |    |   V1 |   V2 |   V3 |   V4 |   V5 |   V6 |   V7 |   V8 |   V9 |
    |---:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|
    |  0 |    2 |    2 |    2 |    2 |    1 |    1 |    2 |    1 |    1 |
    |  1 |    2 |    2 |    2 |    2 |    1 |    1 |    1 |    2 |    1 |
    |  2 |    2 |    2 |    2 |    2 |    1 |    1 |    1 |    1 |    2 |
    |  3 |    2 |    2 |    2 |    2 |    1 |    1 |    1 |    0 |    0 |

    Targets:
    |    |   V10 |
    |---:|------:|
    |  0 |     1 |
    |  1 |     1 |
    |  2 |     1 |
    |  3 |     1 |

    Unique targets:
     [1 0]

    Unknown x_new:
        V1  V2  V3  V4  V5  V6  V7  V8  V9
    0   2   1   0   2   2   1   1   0   0

    Prediction class:
     [1]
        
    Accuracy score:
     1.0
"""