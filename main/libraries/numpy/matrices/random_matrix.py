import numpy as np
from icecream import ic

# Random matrix
np.random.seed(0)
rand_matrix_floats = np.random.random(3) # floats
rand_matrix_ints = np.random.randint(1, 11, 3) # integers

ic(rand_matrix_floats)
ic(rand_matrix_ints)

"""
    ic| rand_matrix_floats: array([0.5488135 , 0.71518937, 0.60276338])
    ic| rand_matrix_ints: array([ 4,  8, 10])
"""