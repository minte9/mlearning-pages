""" Derived function
    f(x) = ax^2

Increment method
    y + Dy = a(x + Dx)^2
    y + Dy = ax^2 + 2axDx + a(Dx^2)
    Dy = 2axDx + a(Dx^2)
    Dy/Dx = 2ax + aDx

At limit Dx -> 0
    f'(x) = 2ax

Derived function f'(x) is pronounced "f prime of x"
"""

from sympy import *

x = Symbol('x')
t = Symbol('t')

a = 1
f = a*x**2
d = f.diff(x)

print(f'f\'(x) = {d}')  # f'(x) = 2*x

s = 16*t**2
d = s.diff(t)

print(f's\'(t) = {d}')  # s'(x) = 32*x