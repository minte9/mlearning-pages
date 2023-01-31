""" Differentiation of Simple Polynomials
Contributions from each term
    y = ax^2 + bx + c
    y' = 2ax + b
"""

from sympy import *
x = Symbol('x')

a = 2
b = 3
c = 4

y = a*x**2 + b*x + c
d = y.diff(x)
print(y)            # function
print(y.diff(x))    # first derivative
print(d.diff(x))    # second derivative
    # 2*x**2 + 3*x + 4
    # 4*x + 3
    # 4