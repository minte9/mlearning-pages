import matplotlib.pyplot as plt
import numpy as np

X = np.linspace(-1, 1, 100) 
I = np.ones(100)

def myfig():
    fig, ax = plt.subplots()
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.grid(True, which='both')
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')
    return fig, ax

# sqrt(x)
fig, ax = myfig()
ax.plot(X, np.sqrt(I + X), label='f(x) = sqrt(x)')   
ax.plot(X, np.sqrt(I - X), label='f(x) = sqrt(1-x)')
plt.legend(loc='upper right')

# x^2
fig, ax = myfig()
ax.plot(X,  X**2, label='f(x) =  x^2')
ax.plot(X, -X**2, label='f(x) = -x^2')
plt.legend(loc='upper right')

# 1/x^2
fig, ax = myfig()
X1 = np.linspace(0.5, 10, 100) 
X2 = np.linspace(-0.5, -10, 100) 
ax.plot(X1, 1/X1**2, label='f(x) =  1/x^2')
ax.plot(X2, -1/X2**2, label='f(x) = -1/x^2')
plt.legend(loc='upper right')

# x^2-4 / x-2
fig, ax = myfig()
X = np.linspace(-4, 5, 10) 
ax.plot(X, (X**2-4) / (X-2), label='f(x) = x^2-4 / x-2') # no value at x=2
ax.plot(X, (X + 2) + 1, label='f(x) = x + 2')
plt.scatter(X, (X**2 - 4) / (X - 2))
plt.legend(loc='upper right')

plt.show()