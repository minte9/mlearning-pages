import numpy as np
import matplotlib.pyplot as plt

def myfig():
    fig, ax = plt.subplots()
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.grid(True, which='both')
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')
    return fig, ax

fig, ax = myfig()

X = np.linspace(-2, 2, 100) 
plt.plot(X, X**2, label='f(x) = x^2') 


def plot_slope(x1, y1, slope):

    # Calculate intercept "b" in y = ax + b
    b = y1 - slope * x1

    # Draw a point around the input point
    x2, y2 = (x1 + 1, slope * (x1 + 1) + b)

    # Draw start line point
    plt.scatter(x1, y1)
    plt.annotate('(%s, %s)'%(x1, y1), xy=(x1, y1), xytext=(x1, y1))

    # Draw slope line
    plt.plot((x1, x2), (y1, y2), linestyle='--', label='d(x) = 2x + %s' %b)


x1, y1 = (-1, 1)
slope = 2 * x1
plot_slope(x1, y1, slope)

x1, y1 = (-1.42, 2)
slope = 2 * x1
plot_slope(x1, y1, slope)

x1, y1 = (-2, 4)
slope = 2 * x1
plot_slope(x1, y1, slope)

plt.legend(loc='upper right')
plt.show()