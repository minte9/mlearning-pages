""" Linear Regression (two parameters)
h(x) = ax + by + c
We can predict the CO2 emission of a car based on the size of the engine, 
but with multiple regression we can throw in more variables, 
like the weight of the car, to make the prediction more accurate
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd
import pathlib
import warnings  # fitted without feature names
warnings.filterwarnings("ignore", category=Warning)

DIR = pathlib.Path(__file__).resolve().parent
with open(DIR / 'data/cars.csv') as file:
    df = pd.read_csv(DIR / 'data/cars.csv')
    X = df[[
        'Weight',
        'Volume',
    ]].values
    y = df['CO2'].values

r = LinearRegression().fit(X, y) 

# Draw surface
fig = plt.figure()
Ax, Ay = np.meshgrid(
    np.linspace(df.Weight.min(), df.Weight.max(), 100),
    np.linspace(df.Volume.min(), df.Volume.max(), 100)
)
onlyX = pd.DataFrame({'Weight': Ax.ravel(), 'Volume': Ay.ravel()})
fittedY = r.predict(onlyX)
fittedY = np.array(fittedY)

ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['Weight'], df['Volume'], df['CO2'], c='g', marker='x', alpha=0.5)
ax.plot_surface(Ax, Ay, fittedY.reshape(Ax.shape), color='b', alpha=0.3)
ax.set_xlabel('Weight')
ax.set_ylabel('Volume')
ax.set_zlabel('CO2')

# Predictions
X = [1600, 1252]   # Honda Civic, 1600, 1252 / CO2: 94
y = r.predict([X]) # CO2: 101.5
print(y.round(1).item())
ax.plot(X[0], X[1], y[0], 'o', color='r')

X = [1200, 780]    # ?
y = r.predict([X]) # CO2: 94.8
print(y.round(1).item())
ax.plot(X[0], X[1], y[0], 's', color='g')

plt.show()