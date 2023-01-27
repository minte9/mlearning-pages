"""
Annotate points on plot
"""

import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0.0, 5.0, 0.01)
y = np.cos(2 * np.pi * x)

fig, ax = plt.subplots(figsize=(5, 2.7))
ax.plot(x, y, lw=2)

ax.annotate('local max', xy=(2, 1), xytext=(3, 1.5),
    arrowprops=dict(facecolor='gray', shrink=0.05))
ax.set_ylim(-2, 2)
plt.show()