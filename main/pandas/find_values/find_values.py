""" Find values and statistics

Pandas has multiple built-in methods for descriptive statistics.
Can be applied to a column or to whole dataframe.
"""

import pandas as pd
import pathlib

DIR = pathlib.Path(__file__).resolve().parent / '../_data/'
df = pd.read_csv(DIR / 'titanic.csv')


# Statistics
T = pd.DataFrame()
T['max'] = [df['Age'].max()]
T['min'] = [df['Age'].min()]
T['avg'] = [df['Age'].mean()]
T['sum'] = [df['Age'].sum()]
T['cnt'] = [df['Age'].count()]
print(T.to_markdown())

# |    |   max |   min |    avg |     sum |   cnt |
# |---:|------:|------:|-------:|--------:|------:|
# |  0 |    71 |  0.17 | 30.398 | 22980.9 |   756 |


# # Unique values
T = pd.DataFrame()
T['unique_sex'] = df['Sex'].unique()
T['value_counts'] = [df['Sex'].value_counts()[0], df['Sex'].value_counts()[1]] 
print(T.to_markdown())

# |    | unique_sex   |   value_counts |
# |---:|:-------------|---------------:|
# |  0 | female       |            851 |
# |  1 | male         |            462 |

T = pd.DataFrame()
T['PClass_value_counts']  = df['PClass'].value_counts()
print(T.to_markdown())

# |     |   PClass_value_counts |
# |:----|----------------------:|
# | 3rd |                   711 |
# | 1st |                   322 |
# | 2nd |                   279 |
# | *   |                     1 |


# Missing values
df = df[df['Age'].isnull()]
print('Missing values | Age Null:')
print(df.head(2).to_markdown())

# |    | Name           | PClass   |   Age | Sex    |   Survived |   SexCode |
# |---:|:---------------|:---------|------:|:-------|-----------:|----------:|
# | 12 | Aubert, Mrs Leo| 1st      |   nan | female |          1 |         1 |
# | 13 | Barkworth, Mr A| 1st      |   nan | male   |          1 |         0 |