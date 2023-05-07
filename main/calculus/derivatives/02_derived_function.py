""" Derivatives / Differentiation

The process of using increment method to get the derivative 
is called differentiation.

Derived function for f(x) = ax^2 is f'(x) and is pronounced "f prime of x"
It means instantaneous rate of change of y with respect to x at value x1

Increment method:
    y + Dy = a(x + Dx)^2
    y + Dy = ax^2 + 2axDx + a(Dx^2)
    Dy = 2axDx + a(Dx^2)
    Dy/Dx = 2ax + aDx

As Dx approaches the limit Dx -> 0, the derived is:
    f'(x) = 2ax
"""

from sympy import *

t = Symbol('t')
s = 16*t**2 # Speed of a falling object
s_derivative = s.diff(t)

r = Symbol('r')
A  = pi*r**2 # Circle area
A_derivative = A.diff(r)

x = Symbol('x')
f  = x**2 # Function f(x)
f_derivative = f.diff(x)

assert 32*t == s_derivative
assert 2*pi*r == A_derivative
assert 2*x == f_derivative

print(f"Falling speed: s = {s}  s' = {s_derivative}")
print(f"Circle area:   A = {A}  A' = {A_derivative}")
print(f"Function:      f ={f}   f' = {f_derivative}")

"""
    Falling speed: s = 16*t**2  s' = 32*t
    Circle area:   A = pi*r**2  A' = 2*pi*r
    Function:      f =x**2      f' = 2*x
"""