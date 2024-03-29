""" Gradient descent Algorithm

1. Initialize the parameters
    select initial set of params for the model

2. Compute the cost function
    differences between predictions and actual values

3. Compute the gradients
    partial derivatives of cost function

4. Update the parameters
    use the gradients and a learning rate

5. Repeat steps 2-4
"""

import numpy as np

def cost(theta, x, y):

    y_pred = np.dot(x, theta)
    error = y_pred - y
    return (1 / (2 * len(y))) * np.dot(error.T, error)

def gradient_descent(x, y, theta, lr, num_iterations):

    cost_history = np.zeros(num_iterations)
    for i in range(num_iterations):

        y_pred = np.dot(x, theta)
        error = y_pred - y

        theta = theta - (lr/len(y)) * np.dot(x.T, error)
        cost_history[i] = cost(theta, x, y)

    return theta, cost_history

x = np.array([[1, 2], [1, 3], [1, 4], [1, 5]])
y = np.array([[7], [6], [5], [7]])

theta = np.random.randn(2, 1)
lr = 0.01
num_iterations = 1000

theta, cost_history = gradient_descent(x, y, theta, lr, num_iterations)
print("Theta: ", theta) 
    # [[4.55230192] [0.43431721]]
