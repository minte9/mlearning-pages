import pandas as pd
import numpy as np

# Set the initial traning data
A = ['apple']*1 + ['orange']*2 + ['banana']*2
B = ['apple']*5 + ['orange']*2 + ['banana']*0

# Probability (by hand)
P1 = [1/5, 2/5, 2/5] 
P2 = [5/7, 2/7, 0/7]

# Probability (with pandas)
A = pd.Series(A)
B = pd.Series(B)
P3 = A.value_counts(normalize=True)
P4 = B.value_counts(normalize=True)

# Entropy (Shannon model)
H3 = -1 * np.sum(P3 * np.log2(P3))
H4 = -1 * np.sum(P4 * np.log2(P4))
assert H3 > H4

# Output results
print("Datasets:")
print("A =", A.values)
print("B =", B.values)

print("\n Probability distributions (by hand):")
print(P1)
print(P2)

print("\n Probability distributions (with pandas):")
print(P3.values)
print(P4.values)

print("\n Entropies:")
print(H3)
print(H4)