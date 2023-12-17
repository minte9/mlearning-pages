""" Data cleaning / Location
Real world cases could have millions of rows and columns.
"""

import pandas as pd
import pathlib

DIR = pathlib.Path(__file__).resolve().parent
df = pd.read_csv(DIR / '../_data/titanic.csv')

# Dispaly 2nd to 4th row
print(df.iloc[1:4])

# Select by index (name)
df = df.set_index(df['Name'])
print(df.loc['Allen, Miss Elisabeth Walton'])

"""

                                            Name PClass   Age     Sex  Survived  SexCode
1                    Allison, Miss Helen Loraine    1st   2.0  female         0        1
2            Allison, Mr Hudson Joshua Creighton    1st  30.0    male         0        0
3  Allison, Mrs Hudson JC (Bessie Waldo Daniels)    1st  25.0  female         0        1

Name        Allen, Miss Elisabeth Walton
PClass                               1st
Age                                 29.0
Sex                               female
Survived                               1
SexCode                                1
Name: Allen, Miss Elisabeth Walton, dtype: object

"""