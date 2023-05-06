""" Operations / Matrix Transpose
Transposing is a common operation in linear algebra.
Indices of column and rows of each element are swapped.
A vector cannot be transposed.
"""

import numpy as np

M = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
])

M_transposed = M.T

assert M[0, 1] == M_transposed[1, 0]
assert M[1, 0] == M_transposed[0, 1]
assert M[1, 1] == M_transposed[1, 1]

print("Matrix =\n", M)
print("Transposed =\n", M_transposed)

"""
    Matrix =
     [[1 2 3]
      [4 5 6]
      [7 8 9]]
    Transposed =
     [[1 4 7]
      [2 5 8]
      [3 6 9]]
"""