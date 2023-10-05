""" Vectorize / Broadcasting
"""

import numpy as np

A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
])

B = np.vectorize(lambda x: x + 100)(A)

print("Matrix: \n", A)
print("Vectorized: \n", B)

B = A + 100

print("Broadcasted: \n", B)

"""
    Matrix: 
     [[1 2 3]
      [4 5 6]
      [7 8 9]]
    Vectorized: 
     [[101 102 103]
      [104 105 106]
      [107 108 109]]
    Broadcasted: 
     [[101 102 103]
      [104 105 106]
      [107 108 109]]
"""
