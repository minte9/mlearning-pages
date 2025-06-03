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

# Predict unknown
x1 = 80
y1 = a*x1 + b

print(f'h(x) = {a}x + {b}')
print(f'h({x1}) = {y1}')

# Draw graphics
fig, ax = plt.subplots()
plt.ylim(0, 140)
plt.xlim(0, 140)

ax.plot(X,  Y,  'x', color='g', label='training data')      # Dataset points
ax.plot(x1, y1, 'o', color='r', label=f'h({x1}) = {y1}')    # Unknown point
ax.plot(X, a*X + b,  label=f'h(x) = {b} + {a}x')            # Function line

plt.legend()
plt.show()

"""
    h(x) = 1.3x + -18.0
    h(80) = 86.0
"""