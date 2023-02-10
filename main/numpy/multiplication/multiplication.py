""" Matrix addition
For element-wise multiplication we use *
"""

import numpy as np

A = np.array([
    [1, 1],
    [2, 2],
])

B = np.array([
    [1, 1],
    [3, 3],
])

C = np.dot(A, B)    # first method
C = A @ B           # second method
E = A * B           # element-wise multiplication

print(C)
    # [4 4]
    # [8 8]
print(E)  
    # [1 1]
    # [6 6]

assert (np.dot(A, A) == (A @ A)) .all()     # passed
assert (E[1, 1] == 6)                       # passed                     