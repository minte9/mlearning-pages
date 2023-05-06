""" Data clearing / Condition and filtering

Conditional selecting and filtering data are common tasks.
Sometinmes you are interseted only of some subset of dataset.
"""

import pandas as pd
import pathlib

DIR = pathlib.Path(__file__).resolve().parent / '../_data/'
df = pd.read_csv(DIR / 'titanic.csv')

females = df[df['Sex'] == 'female']
males_60 = df[(df['Sex'] == 'male') & (df['Age'] >= 60)]

print("Condition, Females only:") ; print(females.head(2).to_markdown(), "\n")
print("Filter | Males age 60:") ; print(males_60.head(2).to_markdown(), "\n")

"""
Condition, Females only:
|    | Name                         | PClass   |   Age | Sex    |   Survived |   SexCode |
|---:|:-----------------------------|:---------|------:|:-------|-----------:|----------:|
|  0 | Allen, Miss Elisabeth Walton | 1st      |    29 | female |          1 |         1 |
|  1 | Allison, Miss Helen Loraine  | 1st      |     2 | female |          0 |         1 | 

Filter | Males age 60:
|    | Name                           | PClass   |   Age | Sex   |   Survived |   SexCode |
|---:|:-------------------------------|:---------|------:|:------|-----------:|----------:|
|  9 | Artagaveytia, Mr Ramon         | 1st      |    71 | male  |          0 |         0 |
| 72 | Crosby, Captain Edward Gifford | 1st      |    70 | male  |          0 |         0 | 
"""