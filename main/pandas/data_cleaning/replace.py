"""
pp39
https://github.com/chrisalbon/sim_data/

Pandas replace is an easy way to find and replace values.
Replace accepts regular expressions.
"""

import pandas as pd
import pathlib

DIR = pathlib.Path(__file__).resolve().parent
FILE = DIR / 'data/titanic.csv'
df = pd.read_csv(FILE)

# Replace on value
R = df['Sex'].replace("female", "Woman")
print(R.head(2))
print()

# Replace multipe values
R = df['Sex'].replace(['female', 'male'], ['Woman', 'Man'])
print(R.head(5))
print()

# Replace in entire dataframe
R = df.replace(1, 'one')
print(R.head(2))
print()

# Regex
R = df.replace(r'1st', 'First', regex=True)
print(R.head(2))
print()