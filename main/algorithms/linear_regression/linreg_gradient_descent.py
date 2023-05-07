""" Linear Regression / Algorithm

The algorithm is used to find the best-fit line that models 
the relationship between x and y.

Gradient descent optimization updates the slope and intercept 
of the line on each iteration of the algorithm.

Gradient descent is an iterative optimization algorithm that starts 
with an initial guess for the slope and the intercept and updates them 
in the direction of steepest descent of the cost function until convergence.

The slope or gradient of a function in (x,y) point is the derivative.
"""

import numpy as np
import matplotlib.pyplot as plt

# Training datasets
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

# --------------------------------------------------------------

# Slope and intercept (line params)
m = 0
b = 0

# Learning params
learning_rate = 0.01
num_iterations = 1000

# Gradient descent optimization
for i in range(num_iterations):

    y_pred = m*x + b
    error = y - y_pred

    m_derivative = -(2/len(x)) * sum(x * error)
    b_derivative = -(2/len(x)) * sum(error)

    m = m - learning_rate * m_derivative
    b = b - learning_rate * b_derivative

# --------------------------------------------------------------

m = round(m, 1)
b = round(b, 1)

# Plot points and the best fit line
fig, ax = plt.subplots()
plt.ylim(0, 10)
plt.xlim(0, 10)

ax.plot(x,  y,  'x', color='g', label='training data')
plt.legend()
ax.plot(x, m*x + b,  label=f'h(x) = {m} + {b}x')
plt.show()

print(f"Best fit line for given data: \n y = {m}x + {b}")

"""
    Best fit line for given data: 
     y = 0.6x + 2.1
"""