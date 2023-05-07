""" Devivatives / Polynomials Differentiation

Differentiation of simple polynomials takes into account
each term:
    y   = ax^2 + bx + c
    y'  = 2ax + b
    y'' = 2a
"""

from sympy import *

x = Symbol('x')
a = Symbol('a')
b = Symbol('b')
c = Symbol('c')

y = a*x**2 + b*x + c
y_derivative = y.diff(x)
y_derivative_2 = y_derivative.diff(x)

assert 2*a*x + b == y_derivative
assert 2*a == y_derivative_2

print("Function:", y)
print("First derivative:", y_derivative)
print("Second derivative:", y_derivative_2)

"""
    Function: a*x**2 + b*x + c
    First derivative: 2*a*x + b
    Second derivative: 2*a
"""