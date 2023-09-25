""" Cost function J(a) visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

# Training Dataset
X = np.array([30, 46, 60, 65, 77, 95]).reshape(6,1)
Y = np.array([31, 30, 80, 49, 70, 118])

# # Define a range of slope values (parameter 'a') to explore
A = np.linspace(-2, 4.5, 13) # 13 values

# Initialize a list to store the Sum of Squared Residuals (SSR) for each 'a'
SSR = []

# Loop through each 'a' value and calculate SSR
for a in A:
    P = []  # predictions
    SR = [] # square residuals
    for i in X:
        P.append(-18 + a*i)
    for i in range(0, len(X)):
        SR.append((Y[i] - P[i])**2)
    SSR.append(np.sum(SR).round())

# Output results
print("SSR(a -18): \n", SSR, "\n")
print("SSR optim:", min(SSR))

# Define a generic cost function SSR(a) = J
def J(a, b=-18):
    J = 0
    for i in range(len(X)): # number of train points
        J += (Y[i] - (a*X[i] + b))**2
    return J

# Create a plot of the cost function J(a, -18) for different 'a' values
fig, ax = plt.subplots()
ax.plot(A, J(A)) # J(a)
for a in A:
    msg ='J(%.1f, -18)' % a
    ax.plot(a, J(a), 'o', label = msg) # Plot points on the cost function curve
plt.xlabel("a")
plt.ylabel("SSR(a)")  
plt.legend()
plt.show()
