""" NUMPY - MATRIX OPERATIONS
-----------------------------
"""

import numpy as np


""" MIN, MAX, AVG """

M = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
])

max = np.max(M)
min = np.min(M)
avg = np.mean(M)

print(f"Max: {max}")  # 9
print(f"Min: {min}")  # 1
print(f"Avg: {avg}")  # 5.0

max = np.max(M, axis=1)  # max in each row
min = np.min(M, axis=1)
avg = np.mean(M, axis=0)

print(f"Max in each row: {max}")  # [3 6 9]
print(f"Min in each row: {min}")  # [1 4 7]
print(f"Avg in each row: {avg}")  # [4. 5. 6.]


""" RESHAPE """

M = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12],  
])

A = M.reshape(2, 6)
B = M.reshape(1, -1)
C = M.flatten()

print(f"Matrix:\n {M}")
print(f"Reshape(2,6):\n {A}")
print(f"Reshape(1,-1):\n {B}")
print(f"Flattern:\n {C}")

"""
    Reshape(2,6):
    [[ 1  2  3  4  5  6]
     [ 7  8  9 10 11 12]]
    Reshape(1,-1):
    [[ 1  2  3  4  5  6  7  8  9 10 11 12]]
    Flattern:
    [ 1  2  3  4  5  6  7  8  9 10 11 12]
"""


""" TRANSPOSE """

M = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
])

print(f"Transpose: {M.T}")

"""
    [[1 4 7]
     [2 5 8]
     [3 6 9]]
"""


""" INVERSE """

M = np.array([
  [4, 3],
  [3, 2],
])

A = np.linalg.inv(M)

I = np.array([  # Identy Matrix
    [1, 0],
    [0, 1]
])

assert (M @ A == I).all()
assert (A @ M == I).all()

print(f"Inverse: {A}")

"""
    [[-2.  3.]
     [ 3. -4.]]
"""



""" ADDITION & SUBSTRACTION """

A = np.array([
    [1, 1],
    [2, 2],
])

B = np.array([
    [1, 1],
    [3, 3],
])

C = A + B

print(f"Addition:\n {A} +\n\n {B} =\n\n {C}")

"""
    [[1 1]
     [2 2]] +

    [[1 1]
     [3 3]] =

    [[2 2]
     [5 5]]
"""


""" MULTIPLICATION """

A = np.array([
    [1, 1],
    [2, 2],
])

B = np.array([
    [1, 1],
    [3, 3],
])

C = A @ B  # OR

E = A * B  # Element-wise multiplication

print(f"Multiplication:\n {A} @\n\n {B} =\n\n {C}")
print(f"Element-wise Multiplication:\n {A} *\n\n {B} =\n\n {E}")

"""
    Multiplication:

    [[1 1]
     [2 2]] @

    [[1 1]
     [3 3]] =

    [[4 4]
     [8 8]
     ]
    Element-wise Multiplication:
    
    [[1 1]
     [2 2]] *

    [[1 1]
     [3 3]] =

    [[1 1]
     [6 6]]
"""