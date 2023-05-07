""" Linear Regression / Algorithm

This algorithm is used to find the best-fit line that models 
the relationship between x and y.

The gradient descent algorithm is used to update the slope and intercept 
of the line on each iteration of the algorithm.
"""

import numpy as np

# Training data
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

# Slope and intercept (line params)
m = 0
b = 0

# Learning params
learning_rate = 0.01
num_iterations = 1000

# Gradient descent
for i in range(num_iterations):

    y_pred = m*x + b
    error = y - y_pred

    m_derivative = -(2/len(x)) * sum(x * error)
    b_derivative = -(2/len(x)) * sum(error)

    m = m - learning_rate * m_derivative
    b = b - learning_rate * b_derivative

m = round(m, 2)
b = round(b, 2)

print(f"Best fit line for given data: \n y = {m}x + {b}")

"""
    Best fit line for given data: 
    y = 0.62x + 2.14
"""