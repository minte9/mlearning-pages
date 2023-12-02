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

# Evaluate the model
P = []                  # Predictions (on training dataset)
R = []                  # Residuals  
SSR = 0                 # Sum of squared residuals

for i in range(len(X)):
    P = np.append(P, -18 + 1.3*X[i])
    R = np.append(R, Y[i] - P[i])
    SSR += R[i] ** 2

print(f'Prediction function: f(x) = {a}x + {b}')
print('Residuals:', R)
print(f'SSR = {SSR.round(2).item()}')

# Draw graphics
fig, ax = plt.subplots()
plt.ylim(0, 140)
plt.xlim(0, 140)

ax.plot(X, Y, 'x', color='g', label='training data')     # Dataset points
ax.plot(X, a*X + b, label=f'h(x) = {b} + {a}x')          # Function line
for i in range(len(X)):                                  # Residuals
    ax.plot([X[i], X[i]], [P[i], Y[i]], '-', color='c')

plt.legend()
plt.show()

"""
    Prediction function: f(x) = 1.3x + -18.0
    Residuals: [10.  -11.8  20.  -17.5 -12.1  12.5]
    SSR = 1248.15
"""