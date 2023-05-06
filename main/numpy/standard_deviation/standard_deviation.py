""" Variance and Standard deviation

Variance is a measure of the spread of the data.
A high variance means that data are spread over a large range.
A low variannce means that data are clustered close together.

Standard deviation is a measure of the spread of the data 
that is more intuitive than variance, as it is expressed 
in the same units as data.
"""

import numpy as np

A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
])

B = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 900],
])

def get_variation(M):
    N = A.size
    mean = np.mean(A)
    variation = (1/N) * np.sum((A - mean)**2) # population variance
    return variation

def get_standard_variation(M):
    return np.sqrt(get_variation(M))
    return

def np_variation(M):
    return np.var(M)

def np_standard_deviation(M):
    return np.std(M)

assert get_variation(A).round(14) == np.var(A).round(14)
assert get_standard_variation(A) == np.std(A)

print("Matrix A= \n", A)
print("Matrix B= \n", B)
print("A Variation =", np.var(A))
print("B Variation =", np.var(B))
print("A Standard deviation = ", np.std(A))
print("B Standard deviation = ", np.std(B))

"""
    Matrix A= 
    [[1 2 3]
     [4 5 6]
     [7 8 9]]
    Matrix B= 
    [[  1   2   3]
     [  4   5   6]
     [  7   8 900]]
    A Variation = 6.666666666666667
    B Variation = 79206.66666666667
    A Standard deviation =  2.581988897471611
    B Standard deviation =  281.43678982440565
"""
