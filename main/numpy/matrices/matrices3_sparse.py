""" Matrices / Sparse matrix

A sparse matrix stores only non-zero elements, for computation savings.
Compress sparce row (CSR) matrices contain indexes of non-zero values.

Example (Netflix movies/users):
  Columns are every movie on Netflix
  Rows are every Netflix user
  Values are how many times a user watched that movie
"""

import numpy as np
from scipy import sparse

# Sparse matrix
M = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 
    [3, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
])
M_sparse = sparse.csr_matrix(M) # CSR matrix

# Random matrix
np.random.seed(0)
M_random1 = np.random.random(3)          # floats
M_random2 = np.random.randint(1, 11, 3)  # integers

print("Sparce matrix: \n", M_sparse)
print("Random matrix of floats: \n", M_random1)
print("Random matrix of integers: \n", M_random2)

"""
	Sparce matrix: 
	   (1, 1)       1
	  (2, 0)        3
	Random matrix of floats: 
	 [0.5488135  0.71518937 0.60276338]
	Random matrix of integers: 
	 [ 4  8 10]
"""