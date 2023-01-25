""" Derived function
    f(x) = ax^2

Derived function f'(x) is pronounced "f prime of x"
Instantaneous rate of change of y with respect to x at value x1

Increment method
    y + Dy = a(x + Dx)^2
    y + Dy = ax^2 + 2axDx + a(Dx^2)
    Dy = 2axDx + a(Dx^2)
    Dy/Dx = 2ax + aDx

At limit Dx -> 0
    f'(x) = 2ax
"""

from sympy import *

x = Symbol('x')
t = Symbol('t')
r = Symbol('r')

a  = 1
f  = a*x**2 # generic
d1 = f.diff(x)

s  = 16*t**2 # speed
d2 = s.diff(t)

A  = pi*r**2 # circle area
d3 = A.diff(r)

print(f'f\'(x) = {d1}')  # f'(x) = 2*x
print(f's\'(t) = {d2}')  # s'(x) = 32*t
print(f'A\'(r) = {d3}')  # A'(r) = 2*pi*r

f = 5*x
assert f.diff(x) == 5