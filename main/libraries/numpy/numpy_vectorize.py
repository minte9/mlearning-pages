""" NUMPAY - VECTORIZE
----------------------
In NumPy, vectorize allows you to apply a normal Python function
to each element of a NumPy array.

It is mainly for convenience and readability, not performance.

Example:
  Create a vectorized function that adds 100 to each element it receives.
  The function is applied to each element of the array A individually
"""

import numpy as np

A = np.array([  # 2D NumPy array (matrix) with integer values
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
])

B = np.vectorize(lambda x: x + 1)(A)

print(B)

"""
[[ 2  3  4]
 [ 5  6  7]
 [ 8  9 10]]
"""

C = A + 100  # broadcasting (different dimensions allowed)

print(C)

"""
[[101 102 103]
 [104 105 106]
 [107 108 109]]
"""

