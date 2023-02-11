"""
pp38
https://github.com/chrisalbon/sim_data/

Conditional selecting and filtering data are common tasks.
Sometinmes you are interseted only of some subset of dataset.
"""

import pandas as pd
import pathlib

DIR = pathlib.Path(__file__).resolve().parent / '../data/'
df = pd.read_csv(DIR / 'titanic.csv')

# Condition
females = df[df['Sex'] == 'female'].head(2)
print(females)
print()

# Filter
females_65 = df[
    (df['Sex'] == 'female') & 
    (df['Age'] >= 65)
]
print(females_65)