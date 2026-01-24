# POLYNOMIAL FEATURES - Example (Degree = 3)
# ------------------------------------------
# PolynomialFeatures expands simple input features into polynomial terms 
# so a Linear model can learn nonlinear patterns.
#
# We start with very simple data:
#       X = [[2],
#            [3]]
#
# Degree = 3 means:
#   create [1, x, x^2, x^3]
# -------------------------

import numpy as np
from sklearn.preprocessing import PolynomialFeatures

# Original input feature (1 feature, 2 samples)
X = np.array([[2],
              [3]])

# Create the transformer: degree = 3
poly = PolynomialFeatures(degree=3)

# Expand X into polynomial features
X_poly = poly.fit_transform(X)

print("Original X:")
print(X)
print("Polynomial-expanded X (degree=3):")
print(X_poly)

"""
Original X:
[[2]
 [3]]
Polynomial-expanded X (degree=3):
[[ 1.  2.  4.  8.]
 [ 1.  3.  9. 27.]]
"""
