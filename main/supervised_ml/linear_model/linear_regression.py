""" Linear Regression (one parameter)
h(x) = ax + b
Finding the line that best fits the data is known as ...
linear regression (one of the most popular tools in statistics)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Training Dataset
X = np.array([30, 46, 60, 65, 77, 95]).reshape(6,1)
Y = np.array([31, 30, 80, 49, 70, 118])


# Learn a prediction function
r = LinearRegression().fit(X, Y)
a = r.coef_[0].round(1)
b = r.intercept_.round(1)

print(f'f(x) = {a}x + {b}') # f(x) = 1.3x - 18

# Predict unknown
x1 = 80
y1 = a*x1 + b
print(f'f({x1}) = {y1}') # f(80) = 86.0


# Draw graphics
fig, ax = plt.subplots()
plt.ylim(0, 140)
plt.xlim(0, 140)

ax.plot(X, Y, 'x', color='g', label='training data') # Draw dataset points
ax.plot(X, a*X + b, label=f'h(x) = {b} + {a}x') # Draw function line
ax.plot(x1, y1, 'o', color='r', label=f'h({x1}) = {y1}') # Draw unknown point

plt.legend(), plt.show()