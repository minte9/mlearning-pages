""" Replace in DataFrame

Pandas replace is an easy way to find and replace values.
Replace accepts regular expressions.
"""

import pandas as pd
import pathlib

DIR = pathlib.Path(__file__).resolve().parent
df = pd.read_csv(DIR / '../titanic.csv')

R1 = df['Sex'].replace("female", "Woman")
R2 = df['Sex'].replace(['female', 'male'], ['Woman', 'Man'])
R3 = df.replace(1, 'one')
R4 = df.replace(r'1st', 'First', regex=True)

print("Replace in one column:"); print(R1.head(2).to_markdown(), "\n")
print("Replace multiple values:"); print(R2.head(5).to_markdown(), "\n")
print("Replace all:"); print(R3.head(2).to_markdown(), "\n")
print("Regex replace:"); print(R4.head(2).to_markdown(), "\n")

"""
Replace in one column:
|    | Sex   |
|---:|:------|
|  0 | Woman |
|  1 | Woman | 

Replace multiple values:
|    | Sex   |
|---:|:------|
|  0 | Woman |
|  1 | Woman |
|  2 | Man   |
|  3 | Woman |
|  4 | Man   | 

Replace all:
|    | Name                         | PClass   |   Age | Sex    | Survived   | SexCode   |
|---:|:-----------------------------|:---------|------:|:-------|:-----------|:----------|
|  0 | Allen, Miss Elisabeth Walton | 1st      |    29 | female | one        | one       |
|  1 | Allison, Miss Helen Loraine  | 1st      |     2 | female | 0          | one       | 

Regex replace:
|    | Name                         | PClass   |   Age | Sex    |   Survived |   SexCode |
|---:|:-----------------------------|:---------|------:|:-------|-----------:|----------:|
|  0 | Allen, Miss Elisabeth Walton | First    |    29 | female |          1 |         1 |
|  1 | Allison, Miss Helen Loraine  | First    |     2 | female |          0 |         1 | 
"""