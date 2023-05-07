""" Gradient descent (two params, a and b)
Algorithm starts with a random value of the parameter a, b
Then, it finds the direction in which the function
descrease faster and takes a step in that direction, then repeat
"""

import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------

# The model (linear)
def predict(X, a, b):
    Y = X*a + b
    return np.round(Y) # f(x) = ax + b

# Cost function
def J(a, b):
    J = np.sum((Y - predict(X, a, b))**2)
    return J

# Derivatives
def dJ(a, b):
    da = np.sum(-2 * X * (Y - predict(X, a, b))) # b fixed
    db = np.sum(-2 * 1 * (Y - predict(X, a, b))) # a fixed
    return da, db

# Gradient descent
def gradient_descent(X, Y, lr=0.00001, loops=1000):
    a = 0
    b = 0
    for i in range(loops):
        da, db = dJ(a, b)
        a = a - lr * da
        for j in range(loops):
            b = b - lr * db
    return round(a, 1), round(b, 1)

# --------------------------------------------------------------

# Train dataset 1
X = np.array([30, 46, 60, 65, 77, 95])
Y = np.array([31, 30, 80, 49, 70, 118])
print("\nLearning 1")

# Learning a,b
a, b = gradient_descent(X, Y)
print('a =', a, ' b =', b)
print('Predictions:', f'f(x) = {a}x + {b}')

# Predictions
x = 33; y = predict(x, a, b); print("f(%s) =" %x, y)
x = 45; y = predict(x, a, b); print("f(%s) =" %x, y)
x = 62; y = predict(x, a, b); print("f(%s) =" %x, y)

fig, ax = plt.subplots()
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.grid(True, which='both')
ax.axhline(y=0, color='k')
ax.axvline(x=0, color='k')

# Draw dataset 1
ax.plot(X, Y, 'x', color='g', label='training data')
ax.plot(X, a*X + b, label=f'f(x) = {b} + {a}x') # line
ax.plot(55, predict(55, a, b), 'o', color='r')
plt.legend(loc='upper right')

# --------------------------------------------------------------

# Train dataset 2
X = np.array([15, 18, 20, 21, 23, 25, 27, 28, 29, 30, 32, 34, 35, 36])
Y = np.array([23, 74, 65, 82, 135, 321, 440, 400, 290, 620, 630, 610, 560, 568])
print("\nLearning 2")

# Learning a,b
a, b = gradient_descent(X, Y)
print('a =', a, ' b =', b)
print('Predictions:', f'f(x) = {a}x + {a}')

x = 20; y = predict(x, a, b); print("f(%s) =" %x, y)
x = 24; y = predict(x, a, b); print("f(%s) =" %x, y)
x = 33; y = predict(x, a, b); print("f(%s) =" %x, y)

# Draw dataset 2
ax.plot(X, Y, 'x', color='g')
ax.plot(X, a*X + b, label=f'f(x) = {b} + {a}x') # line
ax.plot(55, predict(33, a, b), 'o', color='r')
plt.legend(loc='upper right')

plt.show()

"""
    Learning 1
     a = 1.3  b = -17.3
    Predictions: f(x) = 1.3x + -17.3
     f(33) = 26.0
     f(45) = 41.0
     f(62) = 63.0

    Learning 2
     a = 32.9  b = -533.1
    Predictions: f(x) = 32.9x + 32.9
     f(20) = 125.0
     f(24) = 256.0
     f(33) = 553.0
"""