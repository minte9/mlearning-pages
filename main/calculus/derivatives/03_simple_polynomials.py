""" Devivatives / Polynomials

Differentiation of Simple Polynomials adds contributions
from each term:
    y   = ax^2 + bx + c
    y'  = 2ax + b
    y'' = 2a
"""

from sympy import *

x = Symbol('x')
a = 2
b = 3
c = 4

y = a*x**2 + b*x + c
d = y.diff(x)

print("Function:", y)
print("First derivative:", y.diff(x))
print("Second derivative:", d.diff(x))

"""
    Function:           2*x**2 + 3*x + 4
    First derivative:   4*x + 3
    Second derivative:  4
"""