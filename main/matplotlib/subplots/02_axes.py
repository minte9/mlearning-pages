""" Tow ways of using Matplotlib

Explicitly create Figures and Axes, and call methods on them (OOP) and ...
Rely on pyplot to create and manage Figures and Axes, end use functions
"""

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 2, 100)

# OOP style
fix, ax = plt.subplots()
ax.plot(x, x, label='linear')
ax.plot(x, x**2, label='quadratic')
ax.plot(x, x**3, label='cubic')
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.legend()

# Pyplot style
plt.figure(figsize=(5, 2.7), layout='constrained')
plt.plot(x, x, label='linear')
plt.plot(x, x**2, label='quadratic')
plt.plot(x, x**3, label='cubic')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()

plt.show()