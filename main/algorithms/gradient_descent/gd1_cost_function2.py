""" Cost function J(a,b) visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

# Training Dataset
X = np.array([30, 46, 60, 65, 77, 95]).reshape(6,1)
Y = np.array([31, 30, 80, 49, 70, 118])

# Define a generic cost function SSR(a,b) = J
def J(a, b=-18):
    J = 0
    for i in range(len(X)): # number of train points
        J += (Y[i] - (a*X[i] + b))**2
    return J

# Create a 3D plot of the cost function J(a, b)
fig = plt.figure()
ax = fig.add_subplot(1,1,1,projection='3d')
a = np.linspace(-1, 4, 20)
b = np.linspace(-100, 100, 10)
aa, bb = np.meshgrid(a, b)
ax.plot_surface(aa, bb, J(aa, bb)) # Plot the 3D surface of the cost function
ax.view_init(50,-150) # Set the view angle
plt.show()