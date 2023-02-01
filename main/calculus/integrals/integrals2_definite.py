""" Calcutate definite integrals 
    [a, b]
The symbol for integral is a stylish S from sum, 
or summint slices
"""

import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt

# Function to be integrated
def f(x):
    return x**2

def integral_approximation_A(Y, a, b): # numpy
    approx = (b-a) * np.mean(Y)
    return approx

def integral_approximation_B(f, a, b): #scipy
    approx, error = spi.quad(f, a, b)
    return approx


# Integral bounds
a = 0
b = 1

# Function values
range = np.arange(a, b + .0001, .0001)
Y = f(range)

# Integral approximation
approx_A = integral_approximation_A(Y, a, b) # numpy
approx_B = integral_approximation_B(f, a, b) # scipy
print(approx_A) # 0.33335
print(approx_B) # 0.33333333333333337


# Plotting the function
x = np.linspace(0, 1, 100)
y = f(x)

fig, ax = plt.subplots()
ax.plot(x, y, label="f(x) = x^2")
ax.fill_between(x, y, 0, where=(x >= 0) & (x <= 1), color="gray", alpha=0.5)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
plt.show()