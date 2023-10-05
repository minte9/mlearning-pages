""" Operation / Matrix multiplication
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
D = A @ B           # second method
E = A * B           # element-wise multiplication

assert (C == D).all()
assert (E[1, 1] == 6)     

print("A =\n", A)
print("B =\n", B)
print("A @ A =\n", A @ A)
print("A * A =\n", A * A)

"""
    A =
     [[1 1]
      [2 2]]
    B =
     [[1 1]
      [3 3]]
    A @ A =
     [[3 3]
      [6 6]]
    A * A =
     [[1 1]
      [4 4]]
"""