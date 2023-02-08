""" Operations on Matrices

Vectorization is essentialy a for loop and 
oes not increase performance.

By Broadcasting, Numpy allows operation on arrays, 
even if their dimension are not the same.

We can apply operation along the axes (rows or columns).

pp06
"""

import numpy as np


# -----------------------------------------
# Vectorize

matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
])

# Add 100 every element
add_100 = lambda i: i + 100 # create function that add 100
vectorize_add_100 = np.vectorize(add_100) # create vectorized function
vectorized_matrix = vectorize_add_100(matrix) # apply vectorization to all elements

# One line
vectorized_matrix = np.vectorize(lambda i: i + 100)(matrix)
print(vectorized_matrix); print()
   # [[101 102 103]
   #  [104 105 106]
   #  [107 108 109]]


# -------------------------------------------
# Brodcasting

matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
])

matrix = matrix + 100
print(matrix); print()
   # [[101 102 103]
   #  [104 105 106]
   #  [107 108 109]]


# --------------------------------------------
# Max and Average

matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
])

print(np.max(matrix))  # 9
print(np.min(matrix))  # 1
print(np.mean(matrix)) # 5.0

assert np.max(matrix)  == 9     # passed
assert np.min(matrix)  == 1     # passed
assert np.mean(matrix) == 5     # passed

print(np.max(matrix,  axis=1))    # [3 6 9]  # max in each row
print(np.min(matrix,  axis=1))    # [1 4 7]  # min in each row 
print(np.mean(matrix, axis=0))    # [4 5 6]  # average in each column
print()

np.max(matrix,   axis=1) == [3, 6, 9]   # passed
np.min(matrix,   axis=1) == [1, 4, 7]   # passed
np.mean(matrix,  axis=1) == [4, 5, 6]   # passed