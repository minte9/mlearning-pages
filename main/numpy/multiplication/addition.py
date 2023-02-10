""" Matrix addition
Use + - operators
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

C = np.add(A, A)    # first method
C = A + B           # second method

print(C)
    # [2  2]
    # [5  5]

assert (np.add(A, A) == (A + A)) .all()   # passed