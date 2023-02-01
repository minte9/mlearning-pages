""" Free fall

How long it takes an object dropped from a 400ft point 
to reach the surface of the earth?
    t = ?

Galileo obtain a basic physical principle.
An object fall to earth with the same acceleration (in absence of air)
    a = 32 ft/s^2 
    a = 9.8 m/s^2

Acceleration is the instantaneus rate of change of speed, 
with respect to time:
    v' = 32
    v = 32t + C
    v = 32t (object is dropped, zero speed at start)

Instantaneus speed is the rate of change of distance, 
with respect to time:
    s' = 32t
    s = 16t^2 + C
    s = 16t^2

To answer the initial question, 400ft fall:
    400 = 16t^2
    t = [-5, 5]
    t = 5 (because we have downward fall)

So, from acceleration we can find speed, and then distance
    a = 32
    v = a'
    s = a''
"""

import numpy as np
import matplotlib.pyplot as plt
from sympy import *
import scipy.integrate as spi

# Symbolic representation
t = Symbol('t')
a = 32
v = integrate(a, t)
s = integrate(v, t)
print('Acceleration =', a)  # Acceleration = 32
print('Speed =', v)         # Speed = 32*t
print('Distance =', s)      # Distance = 16*t**2


# Distance traveled in 5 seconds
distance = s.subs(t, 5)
print("Distance =", distance) # 400

# Time for 400ft fall
s = s - 400 # s = 16*t**2 - 400
time = solve(s, t) # find the value of t that makes s = 0
print("Time =", time[1]) # 5


# Plotting
t = np.linspace(0, 10)
s = 16*t**2
fig, ax = plt.subplots()
ax.plot(t, s, label="s(t) = 16t^2")
ax.set_xlabel("t")
ax.set_ylabel("s(t)")
ax.legend()

plt.scatter(5, 400, label="s(5) = 400") 
plt.annotate('(5, 400)', xy=(5, 400), xytext=(5, 400))
plt.plot((5, 5), (0, 400), linestyle='--')
plt.plot((0, 5), (400, 400), linestyle='--')
plt.show()