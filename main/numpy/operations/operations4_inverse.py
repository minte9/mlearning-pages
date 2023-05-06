""" Inverse Matrix
Calculate the inverse of a square matrix.
The new matrix A_inv is calculated so that 
  A * A_inv = I
"""

import numpy as np

A = np.array([
  [4, 3],
  [3, 2],
])

I = np.array([
  [1, 0],
  [0, 1],
])

AInv = np.linalg.inv(A)

print(AInv)
    # [-2  3]
    # [ 3 -4]
    
print(A @ AInv)
    # [1 0]
    # [0 1]

assert (A @ AInv == I)  .all()   # passed
assert (AInv @ A == I)  .all()   # passed