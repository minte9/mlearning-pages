""" Create Matrices
Numpy is the foundation of the Python machine learning stack
The main data structure is the multidimensional array
Arrays are zero-indexed, first element index is 0
"""

import numpy as np

# Vector
row = np.array([1, 2, 3]) # vector as a row
column = np.array([ # vector as a column
    [1],
    [4],
    [3],
])
print(row)      # [1 2 3]
print(column)   # [[1] [4] [3]]
print()

# Matrix
matrix = np.array([ # three rows, two columns
    [1, 2],
    [1, 2],
    [1, 2],
])
print(matrix)   # [[1 2] [1 2] [1 2]]
print()

# Describe
matrix = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
])
print(matrix.shape) # (3, 4)
print(matrix.size)  # 12
print(matrix.ndim)  # 2

# Extract
vector = np.array([1, 2, 3, 4, 5, 6])

print(vector[:])    # [1 2 3 4 5 6]
print(vector[:3])   # [1 2 3]
print(vector[3:])   # [4 5 6]
print(vector[-1])   # 6
print()

assert(vector[:]    == [1, 2, 3, 4, 5, 6])  .all() # passed
assert(vector[:3]   == [1, 2, 3])           .all() # passed
assert(vector[3:]   == [4, 5, 6])           .all() # passed
assert(vector[-1]   == 6)                          # passed

matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
])

print(matrix[1, 1])     # 5
print(matrix[:2, :])    # [[1 2 3] [4 5 6]]     # first two rows
print(matrix[:, 1:2])   # [[2] [5] [8]]         # all rows, second column
print()

assert(matrix[1, 1]     == 5)                              # passed
assert(matrix[:2, :]    == [[1, 2, 3], [4, 5, 6]])  .all() # passed
assert(matrix[:, 1:2]   == [[2], [5], [8]])         .all() # passed 