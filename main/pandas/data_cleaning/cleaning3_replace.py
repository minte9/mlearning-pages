""" Replace in DataFrame

Pandas replace is an easy way to find and replace values.
Replace accepts regular expressions.
"""

import pandas as pd
import pathlib

DIR = pathlib.Path(__file__).resolve().parent
df = pd.read_csv(DIR / '../_data/titanic.csv')

# Replace female/male
df['Sex'] = df['Sex'].replace(['female', 'male'], ['Woman', 'Man'])
df['PClass'] = df['PClass'].replace(r'1st', 'First', regex=True)

print(df.head())

"""
                                            Name PClass    Age    Sex  Survived  SexCode
0                   Allen, Miss Elisabeth Walton  First  29.00  Woman         1        1
1                    Allison, Miss Helen Loraine  First   2.00  Woman         0        1
2            Allison, Mr Hudson Joshua Creighton  First  30.00    Man         0        0
3  Allison, Mrs Hudson JC (Bessie Waldo Daniels)  First  25.00  Woman         0        1
4                  Allison, Master Hudson Trevor  First   0.92    Man         1        0
"""