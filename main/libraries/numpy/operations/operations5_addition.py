""" Matrices / Addition and Substraction (+ -)
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

C = np.add(A, B) # first method
D = A + B        # second method

assert (C == D).all()

print("A =\n", A)
print("B =\n", B)
print("A + B =\n", C)

"""
    A =
     [[1 1]
      [2 2]]
    B =
     [[1 1]
      [3 3]]
    A + B = 
     [[2 2]
      [5 5]]
"""
