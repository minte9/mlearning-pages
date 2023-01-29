"""
Collect data, points with x, y values
    x, y = [1, 2, 3], [2, 4, 5]

Choose a line of best fit 
    y = ax + b
    
Calculate the error difference
    error = y_pred - y

Minimize the error, find the best fit 
    use optimization algorithms (gradient descent)
    
Make predictions
    use the line of best fit to make predictions
"""

import numpy as np

x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

m = 0
b = 0
learning_rate = 0.01
num_iterations = 1000

for i in range(num_iterations):
    
    # predict y values
    y_pred = m*x + b

    # calculate error
    error = y_pred - y

    # calculate the partial derivatives
    m_derivative = -(2/len(x)) * sum(x * (y - y_pred))
    b_derivative = -(2/len(x)) * sum(y - y_pred)

    # update the variables
    m = m - learning_rate * m_derivative
    b = b - learning_rate * b_derivative

print("y =", round(m,2), "x + ", round(b,2))
    # y = 0.62 x +  2.14
