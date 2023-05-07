""" Linear Regression / Algorithm

The algorithm is used to find the best-fit line that models 
the relationship between x and y.
Gradient descent optimization updates the slope and intercept 
of the line on each iteration of the algorithm.
"""

import numpy as np

# Training datasets
X = np.array([1, 2, 3, 4, 5])
Y = np.array([2, 4, 5, 4, 5])

class LinearRegression:

    def __init__(self):
        self.coef_ = []
        self.intercept_ = 0

    def fit(self, X_train, Y_train, learning_rate=0.01, num_iterations=1000):
        lr = LinearRegression()

        # -----------------------------------

        x = X_train
        y = Y_train
        m = 0 
        b = 0
        for i in range(num_iterations):
            y_pred = m*x + b
            error = y - y_pred

            m_derivative = -(2/len(x)) * sum(x * error)
            b_derivative = -(2/len(x)) * sum(error)

            m -= learning_rate * m_derivative
            b -= learning_rate * b_derivative

        # -----------------------------------

        lr.coef_.append(m)
        lr.intercept_ = b
        return lr

# Learn a prediction function
r = LinearRegression().fit(X, Y)
m = r.coef_[0].round(1)
b = r.intercept_.round(1)

# Prediction
x1 = 3
y1 = m*x1 + b

print(f"Learned slope: {m:.1f}")
print(f"Learned intercept: {b:.1f}")
print(f"Best fit line: \n y = {m}x + {b}")
print(f"Prediction for x = {x1}: \n y = {y1:.1f}")

"""
    Learned slope: 0.6
    Learned intercept: 2.1
    Best fit line: 
     y = 0.6x + 2.1
    Prediction for x = 3: 
     y = 3.9
"""
