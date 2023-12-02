from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np, sys
import pandas as pd

import pathlib
DIR = pathlib.Path(__file__).resolve().parent

with open(DIR / 'data/real_estate.csv') as file:
    df = pd.read_csv(file)

    # Features
    X = df[[
        'X1 transaction date',
        'X2 house age',
        'X3 distance to the nearest MRT station',
        'X4 number of convenience stores',
        'X5 latitude',
        'X6 longitude',
    ]].values

    # Label
    y = df['Y house price of unit area'].values

# Train the model
r = LinearRegression().fit(X, y)

# Predictions
X1 = [2013.17, 13, 732.85, 0, 24.98, 121.53]     # price: 39 (train data)
X2 = [2013.58, 16.6, 323.69, 6, 24.98, 121.54]   # price: 51 (train data)
X3 = [2013.17, 33, 732.85, 0, 24.98, 121.53]     # ?

print(df.head())
print('Predict training item1, price =', r.predict([X1]).round(1).item())
print('Predict training item2, price =', r.predict([X2]).round(1).item())
print('Predict unknow item, price =',    r.predict([X3]).round(1).item())

"""
		  No  X1 transaction date  X2 house age  ...
	0      1             2012.917          32.0  ...
	1      2             2012.917          19.5  ...
	2      3             2013.583          13.3  ...
	3      4             2013.500          13.3  ...
	4      5             2012.833           5.0  ...

	[5 rows x 8 columns]

	Predict training item1, price = 38.8
	Predict training item2, price = 48.5
	Predict unknow item,    price = 33.4
"""