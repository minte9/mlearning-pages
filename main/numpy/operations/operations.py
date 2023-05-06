""" Min, Max, Mean
We can apply operations along the axes (rows or columns).
"""

import numpy as np

matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
])

print(np.max(matrix))  # 9
print(np.min(matrix))  # 1
print(np.mean(matrix)) # 5.0

assert np.max(matrix)  == 9     # passed
assert np.min(matrix)  == 1     # passed
assert np.mean(matrix) == 5     # passed

print(np.max(matrix,  axis=1))    # [3 6 9]  # max in each row
print(np.min(matrix,  axis=1))    # [1 4 7]  # min in each row 
print(np.mean(matrix, axis=0))    # [4 5 6]  # average in each column
print()

np.max(matrix,   axis=1) == [3, 6, 9]   # passed
np.min(matrix,   axis=1) == [1, 4, 7]   # passed
np.mean(matrix,  axis=1) == [4, 5, 6]   # passed