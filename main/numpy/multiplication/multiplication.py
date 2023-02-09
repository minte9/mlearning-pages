""" Matrix addition
For element-wise multiplication we use *
"""

import numpy as np


# ----------------------------------------
# Addition or substraction

A = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 2],
])

B = np.array([
    [1, 3, 1],
    [1, 3, 1],
    [1, 3, 8],
])

C = np.add(A, B)    # first method
C = A + B           # second method

print(C)
    # [[ 2  4  2]
    #  [ 2  4  2]
    #  [ 2  4 10]]

assert (np.add(A, B) == (A + B)) .all()   # passed
print()


# ---------------------------------------
# Multiplication

A = np.array([
    [1, 1],
    [1, 2],
])

B = np.array([
    [1, 3],
    [1, 2]
])

C = np.dot(A, B)    # first method
D = A @ B           # second method
E = A * B           # element-wise multiplication

print(C) # [[2 5] [3 7]]
print(D) # [[2 5] [3 7]]
print(E) # [[1 3] [1 4]]
print()

assert (np.dot(A, B) == (A @ B)) .all()     # passed
assert (E[1, 1] == 4)                       # passed                     