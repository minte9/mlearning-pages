""" Matrix Reshape

Reshape maintain the data but as different numbers of rows and columns.
The new matrix should have the same size as original matrix

The argument -1 means "as many as needed"
Flatten transform a matrix into a one-dimensional array.
"""

import numpy as np

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

assert matrix.size == A.size  # passed
assert matrix.size == B.size  # passed
assert matrix.size == C.size  # passed