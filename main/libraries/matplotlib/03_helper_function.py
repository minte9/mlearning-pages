"""
Plot over and over again with different data sets
"""

import matplotlib.pyplot as plt
import numpy as np

def myplot(ax, data1, data2, param_dict):
    out = ax.plot(data1, data2, **param_dict)
    return out

data1, data2, data3, data4 = np.random.randn(4, 100)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 2.7))
myplot(ax1, data1, data2, {'marker': 'x'})
myplot(ax2, data3, data4, {'marker': 'o'})
plt.show()