""" Apply a function over all elements
pp53
Despite the temptation to fall back on for loops,
a more Pythonic solution uses pandas' apply method.

It is common to write a function to perform some useful operation, 
like separating first and last names, converting strings to floats.
"""

import pandas as pd
import pathlib

DIR = pathlib.Path(__file__).resolve().parent / '../_data/'
df = pd.read_csv(DIR / 'titanic.csv')


print("First two names uppercased:")
for name in df['Name'][:2]:
    print(name.upper())

# ALLEN, MISS ELISABETH WALTON
# ALLISON, MISS HELEN LORAINE


print("Use list comprehension:")
print([name.upper() for name in df['Name'][:2]])

# ['ALLEN, MISS ELISABETH WALTON', 
#  'ALLISON, MISS HELEN LORAINE']


print("Better, usign pandas' apply")
def uppercase(x):
    return x.upper()
print(df['Name'].apply(uppercase)[:2].to_markdown())

# |    | Name                         |
# |---:|:-----------------------------|
# |  0 | ALLEN, MISS ELISABETH WALTON |
# |  1 | ALLISON, MISS HELEN LORAINE  |