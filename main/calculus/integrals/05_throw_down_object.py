""" Object thrown downward

When the object leaves the hand it already has some velocity
Suppose, for example, that the velocity is 100 ft/sec
    a = 32
    v = 32t + C;  100 = 32 * 0 + C;  C = 100
    v = 32t + 100
    s' = 16t^2 + 100t + C

Let us agree that the distance is measured from the release point
    t = 0; s = 0; C = 0
    s = 16t^2 + 100t

Distance traveled in 5 seconds is 400
    t = 5
    s = 900
"""

import numpy as np
from sympy import *
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ------------------------------------------------------------

t = Symbol('t')
a = 32
v = integrate(a, t) + 100 # Look Hee
s = integrate(v, t)

# Distance after 5 sec
distance1 = s.subs(t, 5)

# Distance after 2.77 sec
distance2 = s.subs(t, 2.77)

# Time for 400ft thrown fall
s = s - 400                    # s = 16*t**2 + 100t
time = solve(s, t)             # resolve equation that makes s = 0

# ------------------------------------------------------------

# Print results
print('Acceleration =', a)          # 32
print('Speed =', v)                 # 32*t + 100
print('Distance =', s)              # 16*t**2 + 100*t
print("Distance traveled after 5 sec =", distance1)       # 900
print("Distance traveled  after 2.77 sec =", distance2)       # 400
print("Time for 400ft thrown fall =", round(time[0], 2))  # 3

# ------------------------------------------------------------

# Plotting
t = np.linspace(0, 5)
s = 16*t**2 + 100*t
fig, ax = plt.subplots()
ax.plot(t, s, label="s(t) = 16t^2 + 100t")
ax.set_xlabel("t")
ax.set_ylabel("s(t)")
ax.legend()

plt.scatter(2.77, 400, label="s(.277) = 400") 
plt.plot((2.77, 2.77), (0, 400), linestyle='--')
plt.plot((0, 2.77), (400, 400), linestyle='--')
plt.show()

# Animation
def update(frame):
    t = np.linspace(0, frame/10)
    s = 16 * t**2 + 100*t
    ax.clear()
    ax.plot(t, s)
    ax.set_xlabel("t")
    ax.set_ylabel("s(t)")
    ax.set_ylim(400, 0)
    ax.set_xlim(0, 5.2)
    ax.set_title("s(t) = 16t^2 + 100t")

    y = 16*(frame/10)**2 + 100*(frame/10)
    ax.scatter(2.77, y, label="s(t) = 400 - (16t^2 + 100t)")
    ax.plot((frame/10, frame/10), (2.77, y), linestyle='--')
    ax.plot((2.77, frame/10), (y, y), linestyle='--')

fig, ax = plt.subplots()
ani = FuncAnimation(fig, update, frames=np.arange(10, 29, 1), repeat=False)
plt.show()

if 1 == 0: # save image
    ani.save('1427_throw_ball.gif', writer='imagemagick', fps=10)

"""
    Acceleration = 32
    Speed = 32*t + 100
    Distance = 16*t**2 + 100*t - 400
    Distance traveled after 5 sec = 900
    Distance traveled  after 2.77 sec = 399.766400000000
    Time for 400ft thrown fall = 2.77
"""