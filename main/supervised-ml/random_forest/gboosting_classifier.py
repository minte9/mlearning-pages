""" Gradient Boostring Classifier (cancer)

In contract to the random forest, gradient boosting works by
building trees in a serial manner, where each tree tries to correct
the mistakes of the previous one.

By default, it uses 100 trees, maxim depth 3 and learning rate 0.1
With the default params we get 100% accuracy on train data, which could 
lead to overfitting. 
We use a stronger pre-prunning by limiting the maximum depth.
"""

import pathlib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

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

# ------------------------------------------------------------

# Default params
model = GradientBoostingClassifier(random_state=0)
model.fit(X1, Y1)
score1 = model.score(X1, Y1)
score2 = model.score(X2, Y2)

# Reduce overfitting
model = GradientBoostingClassifier(random_state=0, max_depth=1) # Look Here
model.fit(X1, Y1)
score3 = model.score(X1, Y1)
score4 = model.score(X2, Y2)

# ------------------------------------------------------------

# Prediction
x_new =  X2.iloc[0] # expect 1
x_new = pd.DataFrame([x_new], columns=X2.columns)
y_pred = model.predict(x_new)[0]
assert y_pred == 1

# Ouput
print("Test Data:"); print(X2, "\n")
print("Encoded:"); print(Y2, "\n")
print("GradientBoostingClassifier(max_depth=3)")
print(" Training set:", score1)
print(" Test set:", round(score2, 2), "\n")
print("GradientBoostingClassifier(max_depth=1)")
print(" Training score:", round(score3, 2))
print(" Test score:", round(score4, 2), "\n")
print("Unknown:"); print(x_new, "\n")
print("Prediction:", y_pred)

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

    GradientBoostingClassifier(max_depth=3)
     Training set: 1.0
     Test set: 0.75

    GradientBoostingClassifier(max_depth=1)
     Training score: 0.9
     Test score: 1.0

    Unknown:
       outlook  temp  humidity  windy
    9        1     2         1      0 

    Prediction: 1
"""