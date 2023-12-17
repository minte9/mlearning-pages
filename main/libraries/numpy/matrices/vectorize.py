import numpy as np

A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
])

"""
  Create a vectorized function that adds 100 to each element it receives
  The function is applied to each element of the array A individually
"""
B = np.vectorize(lambda x: x + 100)(A)

# Output result
print(B)

"""
     [[101 102 103]
      [104 105 106]
      [107 108 109]]
"""
