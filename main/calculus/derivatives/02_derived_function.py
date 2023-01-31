""" Derived function for f(x) = ax^2
f'(x) is pronounced "f prime of x"

It means instantaneous rate of change of y 
with respect to x at value x1.

Increment method
    y + Dy = a(x + Dx)^2
    y + Dy = ax^2 + 2axDx + a(Dx^2)
    Dy = 2axDx + a(Dx^2)
    Dy/Dx = 2ax + aDx

As Dx approaches the limit Dx -> 0, the derived is:
    f'(x) = 2ax
"""

from sympy import *

# Derivative for falling speed s(t) = 16^2
t = Symbol('t')
s  = 16*t**2
d = s.diff(t)
print(d)
    # s'(x) = 32*t

# Derivative for circle area A(r) = pi*R^2
r = Symbol('r')
A  = pi*r**2
d = A.diff(r)
print(d)
    # A'(r) = 2*pi*r

# Derivative for f(x) = ax^2
x = Symbol('x')
a  = 1
f  = a*x**2
d = f.diff(x)
print(d) 
    # f'(x) = 2*x