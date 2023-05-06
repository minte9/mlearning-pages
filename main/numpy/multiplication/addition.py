""" Addition and Substraction (+ -)
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

C = np.add(A, A) # first method
C = A + B        # second method

assert (np.add(A, A) == (A + A)).all()

print("A =\n", A)
print("B =\n", B)
print("C = A + B =", C)

"""
    A =
     [[1 1]
      [2 2]]
    B =
     [[1 1]
      [3 3]]
    C = A + B = [[2 2]
     [5 5]]
"""
