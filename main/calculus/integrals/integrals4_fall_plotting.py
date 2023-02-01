""" Free fall plotting

Galileo obtain a basic physical principle.
An object fall to earth with the same acceleration (in absence of air)
    a = 32 ft/s^2
"""

import numpy as np
import matplotlib.pyplot as plt
from sympy import *
from matplotlib.animation import FuncAnimation

# Plotting
t = np.linspace(0, 10)
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
    ax.scatter(5, 16*(frame/10)**2, label="s(t) = 400 - 16t^2")
    ax.plot((frame/10, frame/10), (5, 16 * (frame/10)**2), linestyle='--')
    ax.plot((5, frame/10), (16*(frame/10)**2, 16*(frame/10)**2), linestyle='--')

fig, ax = plt.subplots()
ani = FuncAnimation(fig, update, frames=np.arange(10, 51, 1), repeat=True)
plt.show()

# ani.save('1427_falling_ball.gif', writer='imagemagick', fps=5)