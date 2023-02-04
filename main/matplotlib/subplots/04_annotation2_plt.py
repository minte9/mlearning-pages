""" Annotations (when using plt.scatter)
"""

import matplotlib.pyplot as plt

xa = [1, 2, 3, 4]
ya = [5, 6, 7, 8]

plt.scatter(xa, ya, color='r', marker='x', label=f'p({xa}) = {ya}')

for i, p in enumerate(zip(xa, ya)):
    plt.annotate(f"({p[0]}, {p[1]})", (p[0]+0.1, p[1]-.1))

plt.legend(loc='best')
plt.show()