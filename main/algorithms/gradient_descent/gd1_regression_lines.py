""" Linear Regression lines with know intercept parameter (b = -18)
"""

import numpy as np
import matplotlib.pyplot as plt

# Training Dataset
X = np.array([30, 46, 60, 65, 77, 95]).reshape(6,1)
Y = np.array([31, 30, 80, 49, 70, 118])

# Define a range of slope values (parameter 'a') to explore
A = np.linspace(-2, 4.5, 13) # 13 values

# Output results
print("Slope range: \n", A)

# Create a plot for the training data and various linear regression lines
fig, ax = plt.subplots()
plt.ylim(0, 140)
plt.xlim(0, 140)

# Plot training data points
ax.plot(X, Y, 'o', color='g', label='training data') 

for i in range(len(A)):
    msg ='f(x) = -18 + %sx' % A[i].round(1)

    # Plot linear regression lines
    ax.plot(X, -18 + A[i]*X, label = msg) 

plt.xlabel("x")
plt.ylabel("f(x)")  
plt.legend()
plt.show()

"""
    Slope range: 
     [-2.         -1.45833333 -0.91666667 -0.375       0.16666667  0.70833333
     1.25        1.79166667  2.33333333  2.875       3.41666667  3.95833333
     4.5       ]
"""