""" Matrices / Create matrix
"""

import numpy as np

# matrix with three rows, four columns

A = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
])

print("Matrix: \n", A)
print("Shape:", A.shape)
print("Size:", A.size)
print("Number of array dimension:", A.ndim)

"""
	Matrix: 
	 [[ 1  2  3  4]
	  [ 5  6  7  8]
	  [ 9 10 11 12]]
	Shape: (3, 4)
	Size: 12
	Number of array dimension: 2
"""