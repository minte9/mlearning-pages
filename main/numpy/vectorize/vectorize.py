""" Operations on Matrices
Vectorization is essentialy a for loop and does 
not increase performance.
"""

import numpy as np

# Vectorize
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
])

# Add 100 every element
add_100 = lambda i: i + 100                     # create function that add 100
vectorize_add_100 = np.vectorize(add_100)       # create vectorized function
vectorized_matrix = vectorize_add_100(matrix)   # apply vectorization to all

# One line code
vectorized_matrix = np.vectorize(lambda i: i + 100)(matrix)
print(vectorized_matrix); print()
   # [[101 102 103]
   #  [104 105 106]
   #  [107 108 109]]


# Brodcasting (different dimensions are allowed)
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