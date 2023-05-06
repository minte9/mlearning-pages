""" Transpose Matrix
Transposing is a common operation in linear algebra
Indices of column and rows of each element are swapped
A vector cannot be transposed
"""

import numpy as np

matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
])

mT = matrix.T
print(mT)
    # [1 4 7]
    # [2 5 8]
    # [3 6 9]

assert matrix[0, 1] == mT[1, 0]  # passed
assert matrix[1, 0] == mT[0, 1]  # passed
assert matrix[1, 1] == mT[1, 1]  # passed