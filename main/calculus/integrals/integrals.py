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

x = Symbol('x')
t = Symbol('t')

# f(x) = 3x + 4
function = 3*x + 2
derived  = function.diff(x)
integral = integrate(derived, x)
print(function, derived, integral) # 3*x + 4 3 3*x

# s(t) = 16t^2
s = 16*t**2
d = s.diff(t)
I = integrate(d, t)
print(s, d, I) # 16*t**2 32*t 16*t**2


# Plotting the function
def f(x):
    return 3*x + 2
x = np.linspace(0, 1, 100)
y = f(x)

fig, ax = plt.subplots()
ax.plot(x, y, label="f(x) = 3x + 4")
ax.fill_between(x, y, 0, where=(x >= 0) & (x <= 100), color="gray", alpha=0.5)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
plt.show()