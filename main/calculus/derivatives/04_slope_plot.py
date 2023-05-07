import matplotlib.pyplot as plt
import numpy as np

# Falling object s(t) = 16t^2
a = 16

X = np.arange(-5, 5, 0.1)
Y = a*(X**2)

# Plot function line
plt.plot(X, Y)

for x in range(2, 6):

    # Instant speeds
    y = a*(x**2)
    m = 2*a*x
    b = y - m*x

    # Plot points
    plt.plot(x, y, 'rx')
    
    # Plot gradients
    X = np.arange(x, x+2, 0.1)
    t = str(x)
    plt.plot(X, m*X + b, label='s(' + t + ') = ' + str(m))

    # Print results
    print(f"Instant speed for x = {x} is {y} with slope {m} and intercept {b}")

# Plot figure
plt.title('s(t) = 16t^2')
plt.xlabel('t (seconds)')
plt.ylabel('s(t)')
plt.grid(True)
plt.legend(loc='upper left')
plt.show()