import numpy as np
import matplotlib.pyplot as plt

# Input training dataset
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

# Initialize variables for the slope (m) and y-intercept (b) of the line
m = 0
b = 0

# Set the learning rate and the number of iterations for gradient descent
learning_rate = 0.01
num_iterations = 1000

# Perform gradient descent to find the best-fit line
for i in range(num_iterations):
    
    # Calculate the predicted values of y (y_pred) based on the current m and b
    y_pred = m*x + b

    # Calculate the error between the predicted values and the actual values
    error = y - y_pred 

    # Calculate the derivatives of the cost function with respect to m and b
    m_derivative = -(2/len(x)) * sum(x * error)
    b_derivative = -(2/len(x)) * sum(error)

    # Update the values of m and b using the gradient descent algorithm
    m = m - learning_rate * m_derivative
    b = b - learning_rate * b_derivative

# Output the equation of the best-fit line
print(f'Best fit line for given data: y = {m}x + {b}')

# Round the values of m and b for clarity
m = round(m, 1)
b = round(b, 1)


# Create a plot to visualize the data and the best-fit line
fig, ax = plt.subplots()
plt.ylim(0, 10)
plt.xlim(0, 10)

# Plot the data points as 'x' markers in green
ax.plot(x,  y,  'x', color='g', label='Training data')
ax.plot(x, m*x + b,  label=f'h(x) = {m} + {b}x')
plt.legend()
plt.show()