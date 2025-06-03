""" Integrals / Gausssian distribution

Gaussian distribution is also called normal distribution.
We calculate integral for gaussian functions using scipy approximation.

The definitive integral of a function gives the total area 
under the curve function over an interval (a, b).
PDF means probability distribution function.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.stats as stats

# ---------------------------------------------------------------------

# Gaussian function
def gaussian(x, mu, sigma):
    return np.exp((-1*(x - mu)**2) / (2 * sigma**2))

# Define mu and sigma
mu, sigma = 0, 1

# Axis for plot
X = np.arange(-5, 5, 0.001)
Y = stats.norm.pdf(X, mu, sigma) # calculates PDF for normal distribution

# Define integral boundaries
a = -3
b = 3

# Integral approximation (area)
approx, error = integrate.quad(
    lambda x: gaussian(x, mu, sigma), a, b
)

# ---------------------------------------------------------------------

print("Gaussian distribution Area =", approx)

plt.plot(X, Y, label="Gaussian distribution")
plt.fill_between(X, Y, 0, where=(X >= a) & (X <= b), color='gray', alpha=0.5,
label=f'I = %s' %round(approx, 4))
plt.legend()
plt.show()

"""
    Gaussian distribution Area = 2.4998608894830947
"""