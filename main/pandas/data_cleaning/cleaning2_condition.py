""" Data clearing / Condition and filtering

Conditional selecting and filtering data are common tasks.
Sometinmes you are interseted only of some subset of dataset.
"""

import pandas as pd
import pathlib

DIR = pathlib.Path(__file__).resolve().parent
df = pd.read_csv(DIR / '../_data/titanic.csv')

females = df[df['Sex'] == 'female']
males_60 = df[(df['Sex'] == 'male') & (df['Age'] >= 60)]

print("Females:", females.size)
print(females.head())

print("Males age 60+:", males_60.size)
print(males_60.head())

"""

Females: 2772
                                            Name PClass   Age     Sex  Survived  SexCode
0                   Allen, Miss Elisabeth Walton    1st  29.0  female         1        1
1                    Allison, Miss Helen Loraine    1st   2.0  female         0        1
3  Allison, Mrs Hudson JC (Bessie Waldo Daniels)    1st  25.0  female         0        1
6               Andrews, Miss Kornelia Theodosia    1st  63.0  female         1        1
8   Appleton, Mrs Edward Dale (Charlotte Lamson)    1st  58.0  female         1        1

Males age 60+: 108
                                Name PClass   Age   Sex  Survived  SexCode
9             Artagaveytia, Mr Ramon    1st  71.0  male         0        0
72    Crosby, Captain Edward Gifford    1st  70.0  male         0        0
103                 Fortune, Mr Mark    1st  64.0  male         0        0
110  Frolicher-Stehli, Mr Maxmillian    1st  60.0  male         1        0
119         Goldschmidt, Mr George B    1st  71.0  male         0        0

"""