import numpy as np
from icecream import ic

# Three rows, four columns matrix
A = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
])

ic(A)
ic(A.shape)
ic(A.size)
ic(A.ndim) # array dimension number

"""
	ic| A: array([[ 1,  2,  3,  4],
				[ 5,  6,  7,  8],
				[ 9, 10, 11, 12]])
	ic| A.shape: (3, 4)
	ic| A.size: 12
	ic| A.ndim: 2
"""