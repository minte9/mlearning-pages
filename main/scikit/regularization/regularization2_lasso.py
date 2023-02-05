""" Regularization (Lasso regression)

Lasso regression (L1) puts a constrain on the sum of absolute weights values, 
different from Ridge regression (L2) who uses the sum of square weights.

L1 and L2 behave the same at the extremes. 
L1 shrikns many coefficients to be exactly 0, producing a sparse model, 
which can be attractive in problems that benefit from features elimination.
"""

from statistics import mode
from sklearn.preprocessing import PolynomialFeatures, scale
from sklearn.linear_model import Ridge, Lasso
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------
# Dataset

X  = [30, 46, 60, 65, 77, 95]   # area (m^2)
y  = [31, 30, 80, 49, 70, 118]  # price (10,000$)
X2 = [17, 40, 55, 57, 70, 85]   # test data
y2 = [19, 50, 60, 32, 90, 110]

#----------------------------------------------------
# Lasso Regression

degree_ = 4
lambda_ = 0.5

# Scale train data to prevent numerical errors
X = np.array(X).reshape(-1, 1)
polyX = PolynomialFeatures(degree=degree_).fit_transform(X)

model1 = Ridge(alpha=lambda_).fit(polyX, y)
model2 = Lasso(alpha=lambda_, max_iter=1300000).fit(polyX, y) # Look Here

t_ = np.array(np.linspace(0, 100, 100)).reshape(-1, 1)
t = PolynomialFeatures(degree=degree_).fit_transform(t_)

# ------------------------------------------------------------------
# Plotting

# Plot train, test data and prediction line
plt.figure(figsize=(6,4))
plt.scatter(X, y, color='blue', label='Training set')
plt.scatter(X2, y2, color='red', label='Test set')

plt.title(f'{degree_}-degree polynomial / Lasso Regression')
plt.plot(t_, model1.predict(t), '--', color='gray', label='Ridge regression')
plt.plot(t_, model2.predict(t), '-', color='orange', label='Lasso regression')

# Predictions
x_unknown = 50
xa = np.array([x_unknown]).reshape(-1,1)

polyX = PolynomialFeatures(degree=degree_).fit_transform(xa)
ya = model1.predict(polyX) # Ridge regression
ya = round(ya[0], 2)

polyX = PolynomialFeatures(degree=degree_).fit_transform(xa)
yb = model2.predict(polyX) # Lasso regression
yb = round(yb[0], 2)

plt.scatter(xa, ya, color='gray', marker='x')
plt.scatter(xa, yb, color='red', marker='x')
plt.annotate(f'({xa[0][0]}, {ya}) - Ridge prediction', (xa+0.1, ya-5))
plt.annotate(f'({xa[0][0]}, {yb}) - Lasso prediction', (xa+0.1, yb+5))

plt.xlim((0, 100))
plt.ylim((0, 130))
plt.legend(loc='upper left')
plt.show()