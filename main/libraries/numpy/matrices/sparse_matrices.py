import numpy as np
from scipy import sparse
from icecream import ic

"""
	Sample matrix (Netfilx movie/users count):
	Every column represents a movie, and every row represents an user
	On values we have how many times each user watched that movie 
	We can see that we have multiple zero values, which is normal.
"""
M = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # User vector
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 
    [3, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
])

# Sparse matrix (CSR)
sparse_matrix = sparse.csr_matrix(M)

ic(sparse_matrix)
print(sparse_matrix)
print("User (index 2) watched the movie (index 0) =", sparse_matrix[2, 0], "times")


"""
ic| sparse_matrix: <3x10 sparse matrix of type '<class 'numpy.int64'>'
                        with 2 stored elements in Compressed Sparse Row format>
  (1, 1)        1
  (2, 0)        3

  User (index 2) watched the movie (index 0) = 3 times
"""