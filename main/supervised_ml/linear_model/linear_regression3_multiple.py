""" Linear Regression (multiple parameters)
h(x) = ax + by + cz + ... 
"""

from os import X_OK
import numpy as np, sys
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd
import pathlib

DIR = pathlib.Path(__file__).resolve().parent
with open(DIR / 'data/real_estate.csv') as file:
    df = pd.read_csv(file)
    X = df[[
        'X1 transaction date',
        'X2 house age',
        'X3 distance to the nearest MRT station',
        'X4 number of convenience stores',
        'X5 latitude',
        'X6 longitude',
    ]].values
    y = df['Y house price of unit area'].values
    r = LinearRegression().fit(X, y)

# Predictions
for X in [
    [2013.17, 13, 732.85, 0, 24.98, 121.53],     # price: 39 (train data)
    [2013.58, 16.6, 323.69, 6, 24.98, 121.54],   # price: 51 (train data)
    [2013.17, 33, 732.85, 0, 24.98, 121.53],     # ?
]:
    print(r.predict([X]).round(1).item())
    # 38.8
    # 48.5
    # 33.4