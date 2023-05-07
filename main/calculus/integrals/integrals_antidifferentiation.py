""" Integrals / Antidifferention

Antidifferention or integration means getting the 
original function from derived function.

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

x = Symbol('x')
t = Symbol('t')

# Symbolic representation
f = 3*x + 2
d  = f.diff(x)
I = integrate(d, x)
print('Derivative =', d)    # Derivative = 3
print('Integral =', I)      # Integral = 3*x
print()

s = 16*t**2
d = s.diff(t)
I = integrate(d, t)
print('Derivative =', d)    # Derivative = 32*t
print('Integral =', I)      # Integral = 16*t**2


# Integral value
def f(x):
    return 3*x + 2
A, err = spi.quad(f, 0, 1) 
print('Integral = ', A)     # Integral = 3.5

def s(t):
    return 16*t**2
A2, err = spi.quad(s, 0, 1) 
print('Integral = ', A2)     # Integral = 5.333333333333334


# Plotting
x = np.linspace(0, 1, 100)
y = f(x)
fig, ax = plt.subplots()
ax.plot(x, y, label="f(x) = 3x + 2")
ax.fill_between(x, y, 0, where=(x >= 0) & (x <= 1), color="gray", alpha=0.5,
label=f'Integral (Area) = %s' %A)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
plt.show()