""" Operations / Min, Max, Mean
"""

import numpy as np

M = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
])

print("Max item =", np.max(M))
print("Max item =", np.min(M))
print("Average =",  np.mean(M))
print("Max in each row =", np.max(M,  axis=1))
print("Min in each row =", np.min(M,  axis=1))
print("Average in each row =", np.mean(M, axis=0))

"""
    Max item = 9
    Max item = 1
    Average = 5.0
    Max in each row = [3 6 9]
    Min in each row = [1 4 7]
    Average in each row = [4. 5. 6.]
"""