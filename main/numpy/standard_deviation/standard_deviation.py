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

# Algorithms
def variation(M):
    N = A.size
    mean = np.mean(A)
    variation = (1/N) * np.sum((A - mean)**2) # population variance
    return variation

def standard_deviation(M):
    return np.sqrt(variation(M))
    return

# Native
def variation_np(M):
    return np.var(M)

def standard_deviation_np(M):
    return np.std(M)

assert variation(A).round(14) == variation_np(A).round(14)
assert standard_deviation(A)  == standard_deviation_np(A)

print("A = \n", A)
print("B = \n", B)
print("np.var(A) variation =", np.var(A))
print("np.var(B) variation =", np.var(B))
print("np.std(A) standard deviation = ", np.std(A))
print("np.std(B) standard deviation = ", np.std(B))

"""
    A = 
    [[1 2 3]
     [4 5 6]
     [7 8 9]]
    B = 
    [[  1   2   3]
     [  4   5   6]
     [  7   8 900]]
    np.var(A) variation = 6.666666666666667
    np.var(B) variation = 79206.66666666667
    np.std(A) standard deviation =  2.581988897471611
    np.std(B) standard deviation =  281.43678982440565
"""
