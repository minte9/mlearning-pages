""" Regularization (Ridge regression - L2)

Basis expansion implies a more complex model.
One way to decrese this complexity is by regularization.

Regularization puts constrains on the sum of weights
in order to keep the weights small.

Ridge regularization uses the sum of square weights, 
which penalizes large weight vectors, and is probably 
the most popular regularized regression method.

Reshape the train data to prevent numerical errors (to large or to small)
By reshaping the data can be transform so that it has a mean of 0 
and a standard deviation of 1

When making predictions data must be transformed using the same 
PolynomialFeatures transformation that was used to preprocess the training data.
"""

from statistics import mode
from sklearn.preprocessing import PolynomialFeatures, scale
from sklearn.linear_model import LinearRegression, Ridge
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------
# Dataset

X  = [30, 46, 60, 65, 77, 95]   # area (m^2)
y  = [31, 30, 80, 49, 70, 118]  # price (10,000$)
X2 = [17, 40, 55, 57, 70, 85]   # test data
y2 = [19, 50, 60, 32, 90, 110]

#----------------------------------------------------
# Ridge Regression

degree_ = 4
lambda_ = 0.5

# Scale train data to prevent numerical errors
X = np.array(X).reshape(-1, 1) # any numbers of rows, one column

polyX = PolynomialFeatures(degree=degree_).fit_transform(X)

model1 = LinearRegression().fit(polyX, y)
model2 = Ridge(alpha=lambda_).fit(polyX, y)

print('Linear coeficients: ', sum(model1.coef_)) # -64.66185222664129
print('Ridge coeficients: ', sum(model2.coef_))  # -7.221838297484756

t_ = np.array(np.linspace(0, 100, 100)).reshape(-1, 1)
t = PolynomialFeatures(degree=degree_).fit_transform(t_)

# ------------------------------------------------------------------
# Plotting

# Plot train, test data and prediction line
plt.figure(figsize=(6,4))
plt.scatter(X, y, color='blue', label='Training set')
plt.scatter(X2, y2, color='red', label='Test set')

plt.title(f'{degree_}-degree polynomial / Ridge Regression')
plt.plot(t_, model1.predict(t), '--', color='gray', label='Linear regression')
plt.plot(t_, model2.predict(t), '-', color='orange', label='Ridge regression')

# Predictions
x_unknown = 50
xa = np.array([x_unknown]).reshape(-1,1)

polyX = PolynomialFeatures(degree=degree_).fit_transform(xa)
ya = model1.predict(polyX) # Linear regression
ya = round(ya[0], 2)

polyX = PolynomialFeatures(degree=degree_).fit_transform(xa)
yb = model2.predict(polyX) # Ridge regression
yb = round(yb[0], 2)

plt.scatter(xa, ya, color='gray', marker='x')
plt.scatter(xa, yb, color='red', marker='x')
plt.annotate(f'({xa[0][0]}, {ya}) - Linear prediction', (xa+0.1, ya-5))
plt.annotate(f'({xa[0][0]}, {yb}) - Ridge prediction', (xa+0.1, yb+5))

plt.xlim((0, 100))
plt.ylim((0, 130))
plt.legend(loc='upper left')
plt.show()