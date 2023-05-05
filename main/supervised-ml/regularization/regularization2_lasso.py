""" Lasso (L1 Regularization)

It puts a constrain on the sum of absolute weights values, it is 
different from Ridge regression (L2) who uses the sum of square weights.

L1 and L2 behave the same at the extremes. 
L1 shrikns many coefficients to be exactly 0, producing a sparse model, 
which can be attractive in problems that benefit from features elimination.
"""

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso
import matplotlib.pyplot as plt
import numpy as np

# Training dataset
X1  = [30, 46, 60, 65, 77, 95]  # area (m^2)
y1  = [31, 30, 80, 49, 70, 118] # price (10,000$)

# Test dataset
X2 = [17, 40, 55, 57, 70, 85]
y2 = [19, 50, 60, 32, 90, 110]

# Lasso Regression
degree_ = 4
lambda_ = 0.8

# Scale train data to prevent numerical errors
X1 = np.array(X1).reshape(-1, 1)
polyX = PolynomialFeatures(degree=degree_).fit_transform(X1)

model1 = Ridge(alpha=lambda_, solver='svd').fit(polyX, y1)
model2 = Lasso(alpha=lambda_, max_iter=1300000).fit(polyX, y1) # Look Here

print('Sum of coeficient (Rigde regularization): ', sum(model1.coef_))
print('Sum of coeficient (Lasso regularization): ', sum(model2.coef_))

t_ = np.array(np.linspace(0, 100, 100)).reshape(-1, 1)
t = PolynomialFeatures(degree=degree_).fit_transform(t_)

# Predictions
x_unknown = 18
xa = np.array([x_unknown]).reshape(-1,1)

polyX = PolynomialFeatures(degree=degree_).fit_transform(xa)
ya = model1.predict(polyX) # Ridge regression
ya = round(ya[0], 2)

polyX = PolynomialFeatures(degree=degree_).fit_transform(xa)
yb = model2.predict(polyX) # Lasso regression
yb = round(yb[0], 2)

# Plot train, test data and prediction line
plt.figure(figsize=(6,4))
plt.scatter(X1, y1, color='blue', label='Training set')
plt.scatter(X2, y2, color='red', label='Test set')

plt.title(f'{degree_}-degree polynomial / Lasso Regression')
plt.plot(t_, model1.predict(t), '--', color='gray', label='Ridge regression')
plt.plot(t_, model2.predict(t), '-', color='orange', label='Lasso regression')

plt.scatter(xa, ya, color='gray', marker='x')
plt.scatter(xa, yb, color='red', marker='x')
plt.annotate(f'({xa[0][0]}) Ridge, price = {ya}', (xa+1.5, ya-5))
plt.annotate(f'({xa[0][0]}) Lasso, price = {yb}', (xa+1.5, yb-5))

plt.xlim((0, 100))
plt.ylim((0, 130))
plt.legend(loc='upper left')

plt.show()

"""
    Sum of coeficient (Rigde regularization):  -4.693509929600461
    Sum of coeficient (Lasso regularization):  0.052552083672473715 # Look Here
"""