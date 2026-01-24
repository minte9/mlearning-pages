# POLYNOMIAL REGRESSION (Degree = 3)
# ----------------------------------
# Goal:
#   Fit a nonlinear relation using LinearRegression on
#   polynomial-expanded features (degree=3).
#
# Steps:
#   1) Define small, hardcoded 1D feature x
#   2) Define target y from a cubic function
#   3) Expand features with PolynomialFeatures(degree=3)
#   4) Fit LinearRegression on the expanded features
#   5) Predict a few values to see it in action
# ---------------------------------------------

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Training data
x = np.array([
    [-3.0],
    [-2.0],
    [-1.0],
    [ 0.0],
    [ 1.0],
    [ 2.0],
    [ 3.0]
])

# Target from a cubic function:
y = 2 + 0.5*x - 1.2*(x**2) + 0.3*(x**3)

# Polynomial expansion to degree 3
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(x)  # columns are [x, x^2, x^3]

# Fit linear regression on the expanded features
model = LinearRegression()
model.fit(X_poly, y)


# Evaluate fit quality (should be R^2 = 1.0 with noiseless data)
y_pred = model.predict(X_poly)
print("R^2 on training data:", round(r2_score(y, y_pred), 4))


# Predictions (include a value not in training)
X_unknown = [1.5]

x = X_unknown[0]
y = y = 2 + 0.5*x - 1.2*(x**2) + 0.3*(x**3)
print("Cubic function:", f"{y:.4f}")

X_val_poly = poly.transform(np.array([X_unknown]))
pred = model.predict(X_val_poly)
print("Prediction:", pred[0])

# --------------------
# Cubic function: 1.06
# Prediction: [1.0625]
