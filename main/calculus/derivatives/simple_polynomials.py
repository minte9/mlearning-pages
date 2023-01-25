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
f = a*x**2 + b*x + c

print(f'f\'(x) = {f.diff(x)}')  # f'(x) = 4*x + 3