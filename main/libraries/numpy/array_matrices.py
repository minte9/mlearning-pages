""" NUMPY - ARRAY (MATRICES)
----------------------------
Numpy main data structure is the multidimensional array.
"""

import numpy as np
from icecream import ic

# Matrix (three rows, four columns)
M = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
])

ic(M)
ic(M.shape)
ic(M.size)
ic(M.ndim)  # array dimension number


"""
ic| row: array([1, 2, 3])
ic| col: array([[1],
                [2],
                [3]])
ic| M.shape: (3, 4)
ic| M.size: 12
ic| M.ndim: 2
"""