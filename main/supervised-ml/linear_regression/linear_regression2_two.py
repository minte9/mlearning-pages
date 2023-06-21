""" Linear Regression  / two parameters
h(x) = ax + by + c

We can predict the CO2 emission of a car based on the size of the engine. 
With multiple regression we can throw in more variables, 
like the weight of the car, to make the prediction more accurate.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd
import pathlib

# fitted without feature names
import warnings  
warnings.filterwarnings("ignore", category=Warning)

# Training Dataset
DIR = pathlib.Path(__file__).resolve().parent
with open(DIR / 'data/cars.csv') as file:
    df = pd.read_csv(DIR / 'data/cars.csv')
    X = df[[
        'Weight',
        'Volume',
    ]].values
    y = df['CO2'].values

# Learn a prediction function
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
X1 = [1600, 1252]    # Honda Civic, 1600, 1252 / CO2: 94
y1 = r.predict([X1]) # CO2: 101.5

X2 = [1200, 780]     # Unknown car
y2 = r.predict([X2]) # CO2: 94.8

print(df, "\n")
print("Honda Civic, 1600, 1252 / CO2:", y1.round(1).item())
print("Unknow car, 1200, 780 / CO2:", y2.round(1).item())

ax.plot(X1[0], X1[1], y1[0], 'o', color='r')
ax.plot(X2[0], X2[1], y2[0], 's', color='g')

plt.show()

"""
    		    Car       Model  Volume  Weight  CO2
	0       Toyoty        Aygo    1000     790   99
	1   Mitsubishi  Space Star    1200    1160   95
	2        Skoda      Citigo    1000     929   95
	3         Fiat         500     900     865   90
	4         Mini      Cooper    1500    1140  105
	5           VW         Up!    1000     929  105
	6        Skoda       Fabia    1400    1109   90
	7     Mercedes     A-Class    1500    1365   92
	8         Ford      Fiesta    1500    1112   98
	9         Audi          A1    1600    1150   99
	10     Hyundai         I20    1100     980   99
	11      Suzuki       Swift    1300     990  101
	12        Ford      Fiesta    1000    1112   99
	13       Honda       Civic    1600    1252   94
	14      Hundai         I30    1600    1326   97
	15        Opel       Astra    1600    1330   97
	16         BMW           1    1600    1365   99
	17       Mazda           3    2200    1280  104
	18       Skoda       Rapid    1600    1119  104
	19        Ford       Focus    2000    1328  105
	20        Ford      Mondeo    1600    1584   94
	21        Opel    Insignia    2000    1428   99
	22    Mercedes     C-Class    2100    1365   99
	23       Skoda     Octavia    1600    1415   99
	24       Volvo         S60    2000    1415   99
	25    Mercedes         CLA    1500    1465  102
	26        Audi          A4    2000    1490  104
	27        Audi          A6    2000    1725  114
	28       Volvo         V70    1600    1523  109
	29         BMW           5    2000    1705  114
	30    Mercedes     E-Class    2100    1605  115
	31       Volvo        XC70    2000    1746  117
	32        Ford       B-Max    1600    1235  104
	33         BMW         216    1600    1390  108
	34        Opel      Zafira    1600    1405  109
	35    Mercedes         SLK    2500    1395  120 

	Honda Civic, 1600, 1252 / CO2: 101.5
	Unknow car, 1200, 780 / CO2: 94.8
"""