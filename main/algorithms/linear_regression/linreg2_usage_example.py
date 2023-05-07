""" Linear Regression / Algorithm

The algorithm is used to find the best-fit line that models 
the relationship between x and y.

Gradient descent optimization updates the slope and intercept 
of the line on each iteration of the algorithm.
"""

import numpy as np

# Algorithm
def linear_regression(x, y, learning_rate=0.01, num_iterations=1000):

    # Initial slope and intercept
    m = 0
    b = 0

    # Gradient descent optimization
    for i in range(num_iterations):
        y_pred = m*x + b
        error = y - y_pred

        # Partial derivatives of loss
        m_derivative = -(2/len(x)) * sum(x * error)
        b_derivative = -(2/len(x)) * sum(error)

        # Update m and b
        m -= learning_rate * m_derivative
        b -= learning_rate * b_derivative

    # Round slope and intercept to 2 decimal places
    m = round(m, 2)
    b = round(b, 2)

    return (m, b)

# Example usage
x_train = np.array([1, 2, 3, 4, 5])
y_train = np.array([2, 4, 5, 4, 5])
x_unknown = 6

# Learn slope and intercept from training data
(m, b) = linear_regression(x_train, y_train)

# Predict y for unknown x
y_pred = m*x_unknown + b

# Print results
print(f"Learned slope: {m:.2f}")
print(f"Learned intercept: {b:.2f}")
print(f"Best fit line: \n y = {m}x + {b}")
print(f"Prediction for x = {x_unknown}: \n y = {y_pred:.2f}")

"""
    Learned slope: 0.62
    Learned intercept: 2.14
    Best fit line: 
     y = 0.62x + 2.14
    Prediction for x = 6: 
    y = 5.86
"""
