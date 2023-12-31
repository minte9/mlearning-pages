"""
Simple line, subplots, bubbles
"""

import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots() # Figure with single axes

# Lines
ax.plot([1, 2, 3, 4], [1, 4, 2, 3]) # [x1, x2, x3, x4], [y1, ...

# Subplots
fig, axs = plt.subplots(1, 2) # Figure with 2x2 grid of axes

# Bubbles
np.random.seed(100000000)
data = {
    'a': np.arange(50),
    'c': np.random.randint(0, 50, 50),
    'd': np.random.randn(50),
}
data['b'] = data['a'] + 10 * np.random.randn(50)
data['d'] = np.abs(data['d']) * 100

fix, ax = plt.subplots(figsize=(5, 2.7))
ax.scatter('a', 'b', c='c', s='d', data=data) # s=size, c=color
ax.set_xlabel('a')
ax.set_ylabel('b')

plt.show()