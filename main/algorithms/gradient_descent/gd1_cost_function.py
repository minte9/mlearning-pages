""" Parameterized SSR (cost function)
Measure the goodness-of-fit (SSR)
For start, let's pretend that intercept is known b = -18
f(x) = ax + -18
SSR(a) = sum(R^2)
"""

import matplotlib.pyplot as plt
import numpy as np

# Training Dataset
X = np.array([30, 46, 60, 65, 77, 95]).reshape(6,1)
Y = np.array([31, 30, 80, 49, 70, 118])

fig, ax = plt.subplots()
plt.ylim(0, 140)
plt.xlim(0, 140)
ax.plot(X, Y, 'o', color='g', label='training data') # points

# Plot lines for some A range values
A = np.linspace(-2, 4.5, 13) # 21 values

for i in range(len(A)):
    msg ='f(x) = -18 + %sx' % A[i].round(1)
    ax.plot(X, -18 + A[i]*X, label = msg) # f(x) = -18 + -2.0x

plt.xlabel("x")
plt.ylabel("f(x)")  
plt.legend()

# Calculate SSR for each a
SSR = []
for a in A:
    P = []  # predictions
    SR = [] # square residuals
    for i in X:
        P.append(-18 + a*i)
    for i in range(0, len(X)):
        SR.append((Y[i] - P[i])**2)
    SSR.append(np.sum(SR).round())

print(SSR)
    # 282654, 197923, 128329, 73872, 34552, 10368, 
    # 1320,
    # 7409, 28635, 64998, 116497, 183133, 264906

# Generic cost function J = SSR(a)
def J(a, b=-18):
    J = 0
    for i in range(len(X)): # number of train points
        J += (Y[i] - (a*X[i] + b))**2
    return J

# Draw J(a)
fig, ax = plt.subplots()
ax.plot(A, J(A)) # J(a)
for a in A:
    msg ='J(%.1f, -18)' % a
    ax.plot(a, J(a), 'o', label = msg) # points
plt.xlabel("a")
plt.ylabel("SSR(a)")  
plt.legend()

# Draw J(a,b)
from mpl_toolkits.mplot3d.axes3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(1,1,1,projection='3d')
a = np.linspace(-1, 4, 20)
b = np.linspace(-100, 100, 10)
aa, bb = np.meshgrid(a, b)
ax.plot_surface(aa, bb, J(aa, bb)) # surface
ax.view_init(50,-150)
plt.show()