""" Antidifferention or Integration
It means getting the original function from derived function.

Integration can be used to find areas, volumes, central points 
and many useful things. 

y' = dy/dx = ax        then y = (ax^2)/2 + C
y  = dy/dx = ax + b    then y = (ax^2)/2 + bx + C
"""

from sympy import *

x = Symbol('x')
t = Symbol('t')

f = x**2
d = f.diff(x)
I = integrate(d, x)
print("Integral for", d, " is", I)
    # Integral for 2*x  is x**2

s = 16*t**2
d = s.diff(t)
I = integrate(d, t)
print("Integral for", d, " is", I)
    # Integral for 32*t  is 16*t**2