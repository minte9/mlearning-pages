""" Matrices / Vectors
Vectors are one-dimensional arrays
"""

import numpy as np

v1 = np.array([1, 2, 3])
v2 = np.array([
    [1],
    [4],
    [3],
])

print("Row vector: \n",     v1)
print("Column vector: \n",  v2)

"""
    Row vector: 
     [1 2 3]
    Column vector: 
    [[1]
     [4]
     [3]]
"""