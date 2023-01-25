""" Differentiation of Simple Polynomials
    y = ax^2 + bx + c

Obtain the contribution from each term
    y' = 2ax + b
"""


from sympy import *

x = Symbol('x')

a = 2
b = 3
c = 4

y = a*x**2 + b*x + c
d = y.diff(x)

print(f'y\' = {d}')  # y' = 4*x + 3