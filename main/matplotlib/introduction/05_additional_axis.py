"""
Plotting data of different magnitude in one chart 
may require an additional y-axis
"""

import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()

x = np.arange(0.0, 5.0, 0.01)
y = np.cos(2 * np.pi * x)

l1, = ax.plot(x, y)
ax2 = ax.twinx() # invisible x-ax
l2, = ax2.plot(x, range(len(x)), 'C1')

plt.show()