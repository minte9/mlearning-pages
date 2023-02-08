""" Matrix Reshape

Reshape maintain the data but as different numbers of
rows and columns.

The new matrix should have the same size as original matrix
The argument -1 means "as many as needed"
Flatten transform a matrix into a one-dimensional array.

Transposing is a common operation in linear algebra.
Indices of column and rows of each element are swapped.
A vector cannot be transposed.

Inverse matrix is calculated so that A * A_inv = I
"""

import numpy as np


# -----------------------------------------------------
# Reshape

matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12],
])

A = matrix.reshape(2, 6)
B = matrix.reshape(1, -1)
C = matrix.flatten()

print(A) # [[ 1  2  3  4  5  6] [ 7  8  9 10 11 12]]
print(B) # [[ 1  2  3  4  5  6  7  8  9 10 11 12]]
print(C) # [  1  2  3  4  5  6  7  8  9 10 11 12]
print()

assert matrix.size == A.size  # passed
assert matrix.size == B.size  # passed
assert matrix.size == C.size  # passed

# --------------------------------------------
# Transpose

matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
])

mT = matrix.T

print(mT) # [[1 4 7] [2 5 8] [3 6 9]]
print()

assert matrix[0, 1] == mT[1, 0]  # passed
assert matrix[1, 0] == mT[0, 1]  # passed
assert matrix[1, 1] == mT[1, 1]  # passed


# -----------------------------------------
# Inverse

A = np.array([
  [4, 3],
  [3, 2],
])

I = np.array([
  [1, 0],
  [0, 1],
])

AInv = np.linalg.inv(A)

print(AInv)      # [[-2  3] [ 3 -4]]
print(A @ AInv)  # [[1 0] [0 1]]

assert (A @ AInv == I)  .all()   # passed
assert (AInv @ A == I)  .all()   # passed