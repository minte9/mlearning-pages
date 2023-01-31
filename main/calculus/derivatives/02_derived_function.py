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

# Falling speed
t = Symbol('t')
s  = 16*t**2
d = s.diff(t)
print("s  =", s) # s  = 16t^2
print("s' =", d) # s' = 32*t

# Circle area
r = Symbol('r')
A  = pi*r**2
d = A.diff(r)
print("A  =", A) # A  = pi*r^2
print("A' =", d) # A' = 2*pi*r

# Function f(x)
x = Symbol('x')
f  = x**2
d = f.diff(x)
print("f  =", f) # f  = ax^2
print("f' =", d) # f' = 2*x