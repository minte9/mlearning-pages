""" Antidifferention or Integration
It means getting the original function from derived function.

Integration can be used to find areas, volumes, central points 
and many other useful things. 

Function: f = 3x + 2
Derived:  d = 3
Integral: I = 3x

y' = ax        then y = (ax^2)/2 + C
y  = ax + b    then y = (ax^2)/2 + bx + C
"""

import numpy as np
import matplotlib.pyplot as plt
from sympy import *
import scipy.integrate as spi

x = Symbol('x')
t = Symbol('t')

f = 3*x + 2
d  = f.diff(x)
I = integrate(d, x)
print(f, d, I) 
    # 3*x + 2 3 3*x

s = 16*t**2
d = s.diff(t)
I = integrate(d, t)
print(s, d, I) 
    # 16*t**2 32*t 16*t**2


# Plotting the Function and Integral
def f(x):
    return 3*x + 2
x = np.linspace(0, 1, 100)
y = f(x)

fig, ax = plt.subplots()
ax.plot(x, y, label="f(x) = 3x + 2")

A, err = spi.quad(f, 0, 1) # Fill the area under the curve
ax.fill_between(x, y, 0, where=(x >= 0) & (x <= 1), color="gray", alpha=0.5, 
    label=f'Area = %s' %A )

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
plt.show()