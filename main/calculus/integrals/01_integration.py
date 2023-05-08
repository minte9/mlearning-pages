""" Integrals / Integration

By antidifferention or integration we can get the original function 
from the derived function.

Integration can be used to find areas, volumes, central points 
and many other useful things. 

Function:   y = 3x + 2
Derivative: d = 3
Integral:   I = 3x

y' = ax        then y = (ax^2)/2 + C
y  = ax + b    then y = (ax^2)/2 + bx + C
"""

import numpy as np
import matplotlib.pyplot as plt
from sympy import *
import scipy.integrate as spi

# -----------------------------------------------------

x = Symbol('x')
t = Symbol('t')

f = 2*x
d = f.diff(x)
d_integration = integrate(d, x)
assert d_integration == f

# -----------------------------------------------------

def func(x):
    return 2*x

# Area with scipy quad()
xa, xb = 0, 2
f_integral, err = spi.quad(func, xa, xb) # lower & upper limits
A = f_integral

# -----------------------------------------------------

fig, ax = plt.subplots()
x = np.linspace(0, 3, 100)
y = func(x)

ax.plot(x, y, label="f(x) = 2x")
ax.fill_between(x, y, 0, where=(x >= xa) & (x <= xb),
    color="gray", alpha=0.5, label=f'Integral (Area) = %s' % A)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
plt.show()

print("Function f(x) =", f)
print("Derivative f' =", d)
print("Integral I =", d_integration)
print('Area A =', A)

"""
    Function   f  = 2*x
    Derivative f' = 2
    Integral   I  = 2*x
    Area       A  = 4.0
"""