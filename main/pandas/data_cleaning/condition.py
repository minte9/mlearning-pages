""" Condition and filtering

Conditional selecting and filtering data are common tasks.
Sometinmes you are interseted only of some subset of dataset.
"""

import pandas as pd
import pathlib

DIR = pathlib.Path(__file__).resolve().parent / '../_data/'
df = pd.read_csv(DIR / 'titanic.csv')


print("Condition, Females only:")
print(df[df['Sex'] == 'female'].head(2).to_markdown())

# |    | Name           | PClass   |   Age | Sex    |   Survived |   SexCode |
# |---:|:---------------|:---------|------:|:-------|-----------:|----------:|
# |  0 | Allen, Miss Eli| 1st      |    29 | female |          1 |         1 |
# |  1 | Allison, Miss H| 1st      |     2 | female |          0 |         1 |


print("Filter | Males age 60:")
print(df[(df['Sex'] == 'male') & (df['Age'] >= 60)].head(2).to_markdown())

# |    | Name             | PClass   |   Age | Sex   |   Survived |   SexCode |
# |---:|:-----------------|:---------|------:|:------|-----------:|----------:|
# |  9 | Artagaveytia, Mr | 1st      |    71 | male  |          0 |         0 |
# | 72 | Crosby, Captain E| 1st      |    70 | male  |          0 |         0 |