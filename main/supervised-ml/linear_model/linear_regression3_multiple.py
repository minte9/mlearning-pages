""" Linear Regression / multiple parameters
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

print(df, '\n')
print('Predict training item1, price =', r.predict([X1]).round(1).item())
print('Predict training item2, price =', r.predict([X2]).round(1).item())
print('Predict unknow item, price =', r.predict([X3]).round(1).item())

"""
		  No  X1 transaction date  X2 house age  ...  X5 latitude  X6 longitude area
	0      1             2012.917          32.0  ...     24.98298     121.54024 37.9
	1      2             2012.917          19.5  ...     24.98034     121.53951 42.2
	2      3             2013.583          13.3  ...     24.98746     121.54391 47.3
	3      4             2013.500          13.3  ...     24.98746     121.54391 54.8
	4      5             2012.833           5.0  ...     24.97937     121.54245 43.1
	..   ...                  ...           ...  ...          ...           ... ...
	409  410             2013.000          13.7  ...     24.94155     121.50381 15.4
	410  411             2012.667           5.6  ...     24.97433     121.54310 50.0
	411  412             2013.250          18.8  ...     24.97923     121.53986 40.6
	412  413             2013.000           8.1  ...     24.96674     121.54067 52.5
	413  414             2013.500           6.5  ...     24.97433     121.54310 63.9

	[414 rows x 8 columns] 

	Predict training item1, price = 38.8
	Predict training item2, price = 48.5
	Predict unknow item,    price = 33.4
"""