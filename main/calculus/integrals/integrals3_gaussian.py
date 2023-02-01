""" Gausssian distribution
Calculate integral for gaussian functions
We use scipy approximation

PDF, probability distribution function 
Gaussian distribution is also called normal distribution

The definitive integral of a function gives
the total area under the curve function over an interval (a, b)
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.stats as stats

# Gaussian function
def gaussian(x, mu, sigma):
    return np.exp((-1*(x - mu)**2) / (2 * sigma**2))

# Define mu and sigma
mu = 0
sigma = 1

# axis for plot
X = np.arange(-5, 5, 0.001)
Y = stats.norm.pdf(X, mu, sigma) # calculates PDF for normal distribution

# Define bounds of integral
a = -3
b = 3

# Integral approximation (area)
approx = integrate.quad(
    lambda x: gaussian(x, mu, sigma), a, b
)
print(approx[0])
    # 2.4998608894830947

plt.plot(X, Y, label="Gaussian distribution")
plt.fill_between(X, Y, 0, where=(X >= a) & (X <= b), color='gray', alpha=0.5)
plt.legend()
plt.show()