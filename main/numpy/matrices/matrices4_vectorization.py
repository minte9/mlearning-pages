""" Vectorization
It is essentialy a for loop that does not increase performance.
When broadcasting different dimensions are allowed.
"""

import numpy as np

M = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
])

# Add 100 every element
add_100 = lambda i: i + 100                 # create function that add 100
vectorize_add_100 = np.vectorize(add_100)   # create vectorized function
M_vectorized1 = vectorize_add_100(M)        # apply vectorization to all

# One line code
M_vectorized2 = np.vectorize(lambda i: i + 100)(M)
assert(M_vectorized1 == M_vectorized2).all()

# Brodcasting
M_vectorized3 = M + 100

print("Matrix: \n", M)
print("Vectorized 100: \n", M_vectorized1)
print("Broadcasted 100: \n", M_vectorized3)

"""
	Matrix: 
	 [[1 2 3]
	  [4 5 6]
	  [7 8 9]]
	Vectorized 100: 
	 [[101 102 103]
	  [104 105 106]
	  [107 108 109]]
	Broadcasted 100: 
	 [[101 102 103]
	  [104 105 106]
	  [107 108 109]]
"""
