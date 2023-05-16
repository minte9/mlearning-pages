""" Decision Tree / Dataset (Play Tennis)
"""

import numpy as np
import pandas as pd
import pathlib
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Dataset
DIR = pathlib.Path(__file__).resolve().parent
df = pd.read_csv(DIR / 'data/play_tennis.csv')
print(df, "\n")

# Encode data
LE = LabelEncoder()
for feature in df.columns:
    df[feature] = LE.fit_transform(df[feature])
print(df)

"""
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
"""
