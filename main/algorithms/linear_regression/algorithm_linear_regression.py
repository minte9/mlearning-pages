""" Linear Regression Algorithm

Linear regression algorithm to find the best-fit line 
that models the relationship between x and y.

The gradient descent algorithm is used to update the slope and intercept 
of the line on each iteration of the algorithm.
"""

import numpy as np

# Training data
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

m = 0 # slope
b = 0 # intercept
learning_rate = 0.01
num_iterations = 1000

# Gradient descent
for i in range(num_iterations):
    
    y_pred = m*x + b
    error = y_pred - y # not used here

    m_derivative = -(2/len(x)) * sum(x * (y - y_pred))
    b_derivative = -(2/len(x)) * sum(y - y_pred)

    m = m - learning_rate * m_derivative
    b = b - learning_rate * b_derivative

m = round(m, 2)
b = round(m, 2)

print(f"Best fit line for given data: \n y = {m}x + {b}")

"""
    Best fit line for given data: 
    y = 0.62x + 0.62
"""