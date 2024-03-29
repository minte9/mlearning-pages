""" Decision Tree / Classifier (Play Tennis)
"""

import pathlib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# Dataset
DIR = pathlib.Path(__file__).resolve().parent
df = pd.read_csv(DIR / 'data/play_tennis.csv')

"""
    outlook,temp,humidity,windy,play
    sunny,hot,high,false,no
    sunny,hot,high,true,no
    overcast,hot,high,false,yes
    rainy,mild,high,false,yes
    rainy,cool,normal,false,yes
    rainy,cool,normal,true,no
    overcast,cool,normal,true,yes
    sunny,mild,high,false,no
    sunny,cool,normal,false,yes
    rainy,mild,normal,false,yes
    sunny,mild,normal,true,yes
    overcast,mild,high,true,yes
    overcast,hot,normal,false,yes
    rainy,mild,high,true,no
"""

# Encode lables
df_encoded = pd.DataFrame()
for col in df.columns:
    df_encoded[col] = LabelEncoder().fit_transform(df[col])

# Train data
X = df_encoded.drop(columns=["play"])
Y = df_encoded['play']

# Fitting the model
dtree_model = DecisionTreeClassifier()
dtree_model.fit(X, Y)

# Prediction
x_unknown =  [1, 0, 1, 0] # expect 1
x_unknown = pd.DataFrame([x_unknown], columns=X.columns)
y_pred = dtree_model.predict(x_unknown)[0]

print("Dataset:"); print(df, "\n")
print("Encoded:"); print(df_encoded, "\n")
print("Sample:"); print(x_unknown, "\n")
print("Prediction:", y_pred)

"""
    Dataset:
        outlook  temp humidity  windy play
    0      sunny   hot     high  False   no
    1      sunny   hot     high   True   no
    2   overcast   hot     high  False  yes
    3      rainy  mild     high  False  yes
    4      rainy  cool   normal  False  yes
    5      rainy  cool   normal   True   no
    6   overcast  cool   normal   True  yes
    7      sunny  mild     high  False   no
    8      sunny  cool   normal  False  yes
    9      rainy  mild   normal  False  yes
    10     sunny  mild   normal   True  yes
    11  overcast  mild     high   True  yes
    12  overcast   hot   normal  False  yes
    13     rainy  mild     high   True   no 

    Encoded:
        outlook  temp  humidity  windy  play
    0         2     1         0      0     0
    1         2     1         0      1     0
    2         0     1         0      0     1
    3         1     2         0      0     1
    4         1     0         1      0     1
    5         1     0         1      1     0
    6         0     0         1      1     1
    7         2     2         0      0     0
    8         2     0         1      0     1
    9         1     2         1      0     1
    10        2     2         1      1     1
    11        0     2         0      1     1
    12        0     1         1      0     1
    13        1     2         0      1     0 
    
    Sample:
       outlook  temp  humidity  windy
    0        1     0         1      0 

    Prediction: 1
"""