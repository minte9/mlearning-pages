""" Operations / Matrix Reshape

Reshape maintain the data but as different numbers of rows and columns.
The new matrix should have the same size as original matrix

The argument -1 means "as many as needed"
Flatten transform a matrix into a one-dimensional array.
"""

import numpy as np

M = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12],
])

A = M.reshape(2, 6)
B = M.reshape(1, -1)
C = M.flatten()

assert M.size == A.size
assert M.size == B.size
assert M.size == C.size

print("Matrix =\n", M)
print("Reshaped (2,6) =\n", A)
print("Reshaped (1,-1) =\n", B)
print("Flatten =\n", C)

"""
	Matrix =
	 [[ 1  2  3]
	  [ 4  5  6]
	  [ 7  8  9]
	  [10 11 12]]
	Reshaped (2,6) =
	 [[ 1  2  3  4  5  6]
	  [ 7  8  9 10 11 12]]
	Reshaped (1,-1) =
	 [[ 1  2  3  4  5  6  7  8  9 10 11 12]]
	Flatten =
	 [ 1  2  3  4  5  6  7  8  9 10 11 12]
"""