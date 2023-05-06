""" Operations / Inverse Matrix

Calculate the inverse of a square matrix.
Matrices are multiplicated using @ (not *)

The new matrix M_inverse is calculated so that 
  M @ M_inverse = I
"""

import numpy as np

M = np.array([
  [4, 3],
  [3, 2],
])

M_inverse = np.linalg.inv(M)

I = np.array([
  [1, 0],
  [0, 1],
])

assert (M @ M_inverse == I).all()
assert (M_inverse @ M == I).all()

print("Matrix =\n", M)
print("Inverse =\n", M_inverse)
print("M @ M_inverse = I: \n", M @ M_inverse)

"""
  Matrix =
   [[4 3]
    [3 2]]
  Inverse =
   [[-2.  3.]
    [ 3. -4.]]
  M @ M_inverse = I: 
   [[1. 0.]
    [0. 1.]]
"""