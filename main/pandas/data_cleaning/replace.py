""" Replace
Pandas replace is an easy way to find and replace values.
Replace accepts regular expressions.
"""

import pandas as pd
import pathlib

DIR = pathlib.Path(__file__).resolve().parent / '../_data/'
df = pd.read_csv(DIR / 'titanic.csv')


# Replace on value
R = df['Sex'].replace("female", "Woman")
print("Replace in one column:")
print(R.head(2).to_markdown())

# |    | Sex   |
# |---:|:------|
# |  0 | Woman |
# |  1 | Woman |


# Replace multipe values
R = df['Sex'].replace(['female', 'male'], ['Woman', 'Man'])
print("Replace multiple values:")
print(R.head(5).to_markdown())

# |    | Sex   |
# |---:|:------|
# |  0 | Woman |
# |  1 | Woman |
# |  2 | Man   |
# |  3 | Woman |
# |  4 | Man   |


# Replace all
R = df.replace(1, 'one')
print("Replace all:")
print(R.head(2).to_markdown())

# |    | Name                         | PClass   |   Age | Sex    | Survived   | SexCode   |
# |---:|:-----------------------------|:---------|------:|:-------|:-----------|:----------|
# |  0 | Allen, Miss Elisabeth Walton | 1st      |    29 | female | one        | one       |
# |  1 | Allison, Miss Helen Loraine  | 1st      |     2 | female | 0          | one       |


# Regex
R = df.replace(r'1st', 'First', regex=True)
print("Regex replace:") 
print(R.head(2).to_markdown())

# |    | Name                         | PClass   |   Age | Sex    |   Survived |   SexCode |
# |---:|:-----------------------------|:---------|------:|:-------|-----------:|----------:|
# |  0 | Allen, Miss Elisabeth Walton | First    |    29 | female |          1 |         1 |
# |  1 | Allison, Miss Helen Loraine  | First    |     2 | female |          0 |         1 |