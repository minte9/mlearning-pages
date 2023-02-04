""" Basis Expansion (Polynomial Models)
Adds non-linear features into to the linear model.

Even though the fifth-degree polynomial model has the lowest
SSR_training, it also has huge SSR_test.

This is a good sign of overfitting.
Another sign of overfitin is by evaluating the coeficients.

Evaluate the weights (sum of coeficients)
The higher the sum, the more the model tends to overfit.
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------
# Dataset

# Train and test
X  = [30, 46, 60, 65, 77, 95]   # area (m^2)
y  = [31, 30, 80, 49, 70, 118]  # price (10,000$)
X2 = [17, 40, 55, 57, 70, 85]
y2 = [19, 50, 60, 32, 90, 110]

# Plot the data
plt.figure(figsize=(6,4))
plt.scatter(X, y, color='blue', label='Training set')
plt.scatter(X2, y2, color='red', label='Test set')
plt.title('Dataset (area, price)')
plt.xlabel('Area (m^2)')
plt.ylabel('Price (10,000$)')
plt.legend(loc='best')

# ----------------------------------------------------
# First-degree polynomial

degrees = 1
p = np.poly1d(np.polyfit(X, y, degrees))
t = np.linspace(0, 100, 100)
print("Model: ", p) # p(t) = 1.303 x - 17.99

plt.figure(figsize=(6,4))
plt.scatter(X, y, color='blue', label='Training set')
plt.scatter(X2, y2, color='red', label='Test set')
plt.plot(t, p(t), color='orange', label=p) # regression line

xa = 50 # unknown
ya = round(p(xa),2) # prediction
plt.scatter(xa, ya, color='r', marker='x')
plt.annotate(f'({xa}, {ya})', (xa+0.1, ya-10))

plt.title('First-degree polinomial')
plt.legend(loc='best')
plt.xlabel('Area (m^2)')
plt.ylabel('Price (10,000$)')
plt.xlim((0, 100))
plt.ylim((0, 130))
plt.show()


# ----------------------------------------------------
# N-degree polynomial

def pred_polinomial(d, x_unknown):

    degrees = d
    p = np.poly1d(np.polyfit(X, y, degrees))
    t = np.linspace(0, 100, 100)
    print(p)
    print(p.coef)

    # Plot train, test data and prediction line
    plt.figure(figsize=(6,4))
    plt.scatter(X, y, color='blue', label='Training set')
    plt.scatter(X2, y2, color='red', label='Test set')
    plt.plot(t, p(t), color='orange') # regression line

    # Evaluate the model (sum of residuals)
    SSR = sum((p(X) - y) ** 2).round()
    SSR2 = sum((p(X2) - y2) ** 2).round()

    # Evaluate the weight (sum of coeficients)
    # The hie
    w = round(sum(abs(p.coef)))
    print(w)

    # plot prediction
    xa = x_unknown
    ya = round(p(xa),2)
    plt.scatter(xa, ya, color='r', marker='x')
    plt.annotate(f'({xa}, {ya}, SSR = {SSR}) SSR2 = {SSR2})', (xa+0.1, ya-10))

    plt.title(f'{d}-degree polynomial')
    plt.legend(loc='best')
    plt.xlabel('Area (m^2)')
    plt.ylabel('Price (10,000$)')
    plt.xlim((0, 100))
    plt.ylim((0, 130))
    plt.show()

pred_polinomial(2, 50) # second-degree polynomial
pred_polinomial(3, 50) # third-degree polynomial
pred_polinomial(4, 50) # fourth-degree polynomial
pred_polinomial(5, 50) # fift-degree polynomial