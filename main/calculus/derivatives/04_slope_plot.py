import numpy as np
import matplotlib.pyplot as plt

# Falling object s(t) = 16t^2
a = 16
X = np.arange(-5, 5, 0.1)
Y = a*(X**2)

for t in range(2, 6):
    y = a*(t**2)
    m = 2*a*t
    b = y - m*t

    print(f"Instant speed for t = {t} is {m}")
    plt.plot(t, y, 'rx') # points

    T = np.arange(t, t+2, 0.1)
    plt.plot(T, m*T + b, label=f"s({t}) = {m}") # gradients

plt.title('s(t) = 16t^2')
plt.xlabel('t (seconds)')
plt.ylabel('s(t)')
plt.grid(True)
plt.plot(X, Y) # function line
plt.legend(loc='upper left')
plt.show()