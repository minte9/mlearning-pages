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

print(np.var(A))    # 6.666666666666667
print(np.std(A))    # 2.581988897471611
print()

print(np.var(B))    # 79206.66666666667
print(np.std(B))    # 281.43678982440565
print()

# Variation mathematics

N = A.size
mean = np.mean(A)

variation = (1/N) * np.sum((A - mean)**2) # population variance
standard_deviation = np.sqrt(variation)

print(variation)            # 6.666666666666666
print(standard_deviation)   # 2.581988897471611

assert standard_deviation == np.std(A)              # passed
assert variation.round(14) == np.var(A).round(14)   # passed