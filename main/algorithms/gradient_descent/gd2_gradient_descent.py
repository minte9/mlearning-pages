""" Gradient descent algorithm
Find the optimal value of a linear regression parameter 'a' for a given dataset
"""

import matplotlib.pyplot as plt
import numpy as np

# Training Dataset
X = np.array([30, 46, 60, 65, 77, 95]).reshape(6,1)
Y = np.array([31, 30, 80, 49, 70, 118])

# Cost function
def J(a):
    J = 0

    # Loop through each data point
    for i in range(len(X)):

        # Calculate the squared error
        J += (Y[i] - (a*X[i] + -18))**2 

    return J

# Derivative of the cost function
def dJ(a):
    dJ = 0
    for i in range(len(X)):

        # Calculate the derivative
        dJ += -2*X[i]*(Y[i] - (a*X[i] + -18)) # d(x^2) = 2x

    return dJ.item()

# Gradient descent
def gradient_descent(X, Y, b=-18, lr=0.00001, loops=15):
    a = 0
    for i in range(15):

        # Update 'a' using the gradient of the cost function
        d = dJ(a)
        a = a - d*lr
        
        print(f'Step {i+1} a = {round(a, 5)}')
    return round(a, 5)

# Result
optim_a = gradient_descent(X, Y)

# Compute values to print and plot
a = 0       # start value
l = 0.00001 # learning rate

a0 = 0
a1 = a  - l * dJ(a)  # step 1
a2 = a1 - l * dJ(a1) # step 2
a3 = a2 - l * dJ(a2) # step 3

# Plot lines SSR curve
fig, ax = plt.subplots()
A = np.linspace(-2, 4.5, 23) # 21 values
ax.plot(A, J(A), label='J(a) = sum(R(X)^2)') # J(a)

# Mark the minimum SSR(a) (optim_a)
ax.plot(optim_a, J(optim_a), 'o', color='g', label='optim_a = 1.3029')

# Draw points (as gradient descends)
ax.plot(a0, J(0), 'o', color='r')
ax.plot(a1, J(a1), 'o', color='r')
ax.plot(a2, J(a2), 'o', color='r')
ax.plot(a3, J(a3), 'o', color='r')

# Draw lines to minimum
ax.plot([a0,  a1], [J(0), J(a1)], color='r')
ax.plot([a1, a2], [J(a1), J(a2)], color='r')
ax.plot([a2, a3], [J(a2), J(a3)], color='r')

# Customize the plot
plt.xlim(-2, 5)
plt.ylim(-10000, 70000)
plt.xlabel("a")
plt.ylabel("SSR(a)")  
ax.axhline(y=0, color='k')
ax.axvline(x=0, color='k')
plt.legend()

# Show the plot
plt.show()

# Print results
print('Derivative of cost function J(0) = ', dJ(0))
print('Step 1 a =', round(a1, 5))
print('Step 2 a =', round(a2, 5))
print('Step 3 a =', round(a3, 5), "\n")
print("Gradient descent optim_a slope: \n", round(optim_a, 4))

"""
    Step 1 a = 0.67218
    Step 2 a = 0.99758
    Step 3 a = 1.15511
    Step 4 a = 1.23137
    Step 5 a = 1.26829
    Step 6 a = 1.28616
    Step 7 a = 1.29481
    Step 8 a = 1.299
    Step 9 a = 1.30102
    Step 10 a = 1.30201
    Step 11 a = 1.30248
    Step 12 a = 1.30271
    Step 13 a = 1.30282
    Step 14 a = 1.30288
    Step 15 a = 1.3029
"""