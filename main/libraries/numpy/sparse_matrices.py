""" NUMPY - SPARCE MATRICES
----------------------------
A sparse matrix stores only non-zero elements, for computation savings.
Compress sparce row (CSR) matrices contain indexes of non-zero values.

Sample: 
Netfilx movie/users count:
    - Every column represents a movie
    - Every row represents an user.

On values we have how many times each user watched that movie.
We can see that we have multiple zero values, which is normal.
"""

import numpy as np
from scipy import sparse

M = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # User vector
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 
    [3, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
])

sparse_matrix = sparse.csr_matrix(M)

print(sparse_matrix)

#  (1, 1)        1
#  (2, 0)        3

print("User[2] watched the movie[0]: ", sparse_matrix[2, 0])

# User[2] watched the movie[0]:  3