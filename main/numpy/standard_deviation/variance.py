import numpy as np

A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
])
B = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 900], # Look Here
])

# Algorithm
def population_variance(X):
    N = X.size
    avg = np.mean(X)
    variance = (1/N) * np.sum((X - avg)**2)
    return variance

A_variance = population_variance(A)
B_variance = population_variance(B)

assert B_variance > A_variance
print("A_variance = ", A_variance.round(2))
print("B_variance = ", B_variance.round(2))
print("np.var(A) = ",  np.var(A).round(2)) # build-in
print("np.var(B) = ",  np.var(B).round(2))

"""
    A_variance =  6.67
    B_variance =  79206.67
    np.var(A) =  6.67
    np.var(B) =  79206.67
"""