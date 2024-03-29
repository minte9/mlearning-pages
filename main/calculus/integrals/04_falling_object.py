""" Object Free fall

What is the time in which an object dropped from a 400ft point 
reaches earth (in the absence of air)
    t = ?

Galileo: An object is falling to Earth with the same acceleration:
    a = 32 ft/s^2 
    a = 9.8 m/s^2

Acceleration is the instantaneus rate of change of speed / time
    v' = 32
    v = 32t + C
    v = 32t     # object is dropped, zero speed at start

Instantaneus speed is the rate of change of distance / time
    s' = 32t
    s = 16t^2 + C
    s = 16t^2

To answer the initial question, 400ft falling object
    400 = 16t^2
    t = [-5, 5]  # both -5 and 5 are correct
    t = 5        # we choose 5, because we have downward fall

By antidifferentiation we can proceed from acceleration to speed, 
and from acceleration to the distance traveled.
    a = 32
    v = a'
    s = a''

Distance traveled in 5 seconds is 400
    t = 5
    s = 400
"""

import numpy as np
from sympy import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ------------------------------------------------------------

t = Symbol('t')
a = 32
v = integrate(a, t)
s = integrate(v, t)

# Distance traveled in 5 seconds
d = s.subs(t, 5)

# Time for 400ft fall
s = s - 400             # s = 16*t**2 - 400
time = solve(s, t)      # resolve equation that makes s = 0

# ------------------------------------------------------------

# Print results
print('Acceleration =', a)  # Acceleration = 32
print('Speed =', v)         # Speed = 32*t
print('Distance =', s)      # Distance = 16*t**2
print("Distance traveled in 5 seconds =", d) # 400
print("Time for 400ft fall =", time[1]) # 5

# Plotting
t = np.linspace(0, 6)
s = 16*t**2
fig, ax = plt.subplots()
ax.plot(t, s, label="s(t) = 16t^2")
ax.set_xlabel("t")
ax.set_ylabel("s(t)")
ax.legend()

plt.scatter(5, 400, label="s(5) = 400") 
plt.plot((5, 5), (0, 400), linestyle='--')
plt.plot((0, 5), (400, 400), linestyle='--')
plt.show()

# Animation
def update(frame):
    t = np.linspace(0, frame/10)
    s = 16 * t**2
    ax.clear()
    ax.plot(t, s)
    ax.set_xlabel("t")
    ax.set_ylabel("s(t)")
    ax.set_ylim(400, 0)
    ax.set_xlim(0, 5.2)
    ax.set_title("s(t) = 16t^2")

    y = 16*(frame/10)**2
    ax.scatter(5, y, label="s(t) = 400 - 16t^2")
    ax.plot((frame/10, frame/10), (5, y), linestyle='--')
    ax.plot((5, frame/10), (y, y), linestyle='--')

fig, ax = plt.subplots()
ani = FuncAnimation(fig, update, frames=np.arange(10, 51, 1), repeat=True)
plt.show() 

if 1 == 0: # save image
    ani.save('1427_falling_ball.gif', writer='imagemagick', fps=10)

"""
    Acceleration = 32
    Speed = 32*t
    Distance = 16*t**2 - 400
    Distance traveled in 5 seconds = 400
    Time for 400ft fall = 5
"""