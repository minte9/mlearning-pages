""" Statistics / Group By
"""

import pandas as pd
import pathlib

DIR = pathlib.Path(__file__).resolve().parent
df = pd.read_csv(DIR / '../_data/titanic.csv')

A = df.groupby('Sex').count()
print(A)

A = df.groupby('Sex').mean(numeric_only=True)
print(A)

A = df.groupby('Sex')['Survived'].count()
print(A)

A = df.groupby(['Sex', 'Survived']).mean(numeric_only=True)
print(A)

"""

Name  PClass  Age  Survived  SexCode
Sex                                         
female   462     462  288       462      462
male     851     851  468       851      851

              Age  Survived  SexCode
Sex                                 
female  29.396424  0.666667      1.0
male    31.014338  0.166863      0.0

Sex
female    462
male      851
Name: Survived, dtype: int64

                       Age  SexCode
Sex    Survived                    
female 0         24.901408      1.0
       1         30.867143      1.0
male   0         32.320780      0.0
       1         25.951875      0.0

"""