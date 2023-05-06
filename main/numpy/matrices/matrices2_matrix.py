""" Matrices / Create matrix

    Numpy is the foundation of the Python machine learning stack
    The main data structure is the multidimensional array

    Arrays are zero-indexed, first element index is 0
    Use ':' to select everything 'up to' or 'after'
"""

import numpy as np

M = np.array([ # three rows, four columns
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
])

v = np.array([1, 2, 3, 4, 5, 6])

print("M items: \n", M)
print("M.shape =", M.shape)
print("M.size =", M.size)
print("M.ndim =", M.ndim, '\n')

print("v[:]  =", v[:])
print("v[:3] =", v[:3])
print("v[3:] =", v[3:])
print("v[-1] =", v[-1], "\n")

print("M[1, 1] second row, second column:", M[1, 1])
print("M[:2, :] up to 2 rows, all columns: \n", M[:2, :])
print("M[:, 1:2] all rows, second column: \n", M[:, 1:2])

"""
	M items: 
	 [[ 1  2  3  4]
	 [ 5  6  7  8]
	 [ 9 10 11 12]]
	M.shape = (3, 4)
	M.size = 12
	M.ndim = 2 

	v[:]  = [1 2 3 4 5 6]
	v[:3] = [1 2 3]
	v[3:] = [4 5 6]
	v[-1] = 6 

	M[1, 1] second row, second column: 6
	M[:2, :] up to 2 rows, all columns: 
	 [[1 2 3 4]
	  [5 6 7 8]]
	M[:, 1:2] all rows, second column: 
	 [[ 2]
	  [ 6]
	  [10]]
"""