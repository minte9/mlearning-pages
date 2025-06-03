""" Apply a function over all elements

Despite the temptation to fall back on for loops,
a more Pythonic solution uses pandas' apply method.

It is common to write a function to perform some useful operation, 
like separating first and last names, converting strings to floats.
"""

import pandas as pd
import pathlib

DIR = pathlib.Path(__file__).resolve().parent
df = pd.read_csv(DIR / '../titanic.csv')

def loop(data):
    for x in data:
        print(x.upper())

def list_comprehension(data):
        print([x.upper() for x in data])

def apply(data):
    def uppercase(x):
        return x.upper()
    print(df['Name'].apply(uppercase)[:2].to_markdown()) # Look Here

print("First two names uppercased:")
print("Loop:"); loop(df['Name'][:2])
print("Use list comprehension:"); list_comprehension(df['Name'][:2])
print("Use pandas' apply() - better!"); apply(df['Name'][:2])


"""
First two names uppercased: 

Loop:
 ALLEN, MISS ELISABETH WALTON
 ALLISON, MISS HELEN LORAINE

Use list comprehension:
 ['ALLEN, MISS ELISABETH WALTON', 'ALLISON, MISS HELEN LORAINE']

Use pandas' apply() - better!
|    | Name                         |
|---:|:-----------------------------|
|  0 | ALLEN, MISS ELISABETH WALTON |
|  1 | ALLISON, MISS HELEN LORAINE  |
"""