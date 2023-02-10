""" Sparse matrices

Sparse matrices only store non-zero elements, for computation savings.
Compress sparce row (CSR) matrices contain indices of non-zero values.

Netflix movies/users example.
Columns are every movie on Netflix
Rows are every Netflix user
Values are how many times a user watched that movie
"""

import numpy as np
from scipy import sparse

matrix = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 
    [3, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
])
matrix_sparse = sparse.csr_matrix(matrix) # CSR matrix
print(matrix_sparse)
    # (1, 1)  1
    # (2, 0)  3
print()

assert matrix_sparse[1, 1] == 1
assert matrix_sparse[2, 0] == 3


# Random
np.random.seed(0)
R1 = np.random.random(3)          # generate floats
R2 = np.random.randint(1, 11, 3)  # generate integers

print(R1) # [0.5488135  0.71518937 0.60276338]
print(R2) # [4  8 10]