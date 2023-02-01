""" Calcutate definite integrals 
    [a, b]
The symbol for integral is a stylish S from sum, 
or summint slices
"""

import numpy as np
import scipy.integrate as spi

def f(x):
    return x**2

def integral_approximation(f, a, b):
    return (b-a) * np.mean(f)

# Integral bounds
a = 0
b = 1

# Function values
range = np.arange(a, b + .0001, .0001)
F = f(range)

# Using numpy and range values
approx = (b-a) * np.mean(F)
print(approx) 
    # 0.33335

# Using scipy
result, error = spi.quad(f, a, b)
print(result) 
    # 0.33333333333333337