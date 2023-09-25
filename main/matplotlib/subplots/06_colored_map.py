"""
Third dimension in a plot represented by a colors in a colormap
"""

import matplotlib.pyplot as plt
import numpy as np

X, Y = np.meshgrid(np.linspace(-3, 3, 128), np.linspace(-3, 3, 128))
Z = (1 - X/2 + X**5 + Y**3) * np.exp(-X**2 - Y**2)

fig, ax = plt.subplots()

co = ax.contourf(X, Y, Z, levels=np.linspace(-1.25, 1.25, 11))
fig.colorbar(co)
ax.set_title('contourf()')
ax.set_xlabel('X')
ax.set_ylabel('Y')

plt.show()