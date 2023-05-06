""" Find values and statistics

Pandas has multiple built-in methods for descriptive statistics.
Can be applied to a column or to whole dataframe.
"""

import pandas as pd
import pathlib

DIR = pathlib.Path(__file__).resolve().parent / '../_data/'
FILE = DIR / 'titanic.csv'
df = pd.read_csv(FILE)

# Statistics
T = pd.DataFrame()
T['max'] = [df['Age'].max()]
T['min'] = [df['Age'].min()]
T['avg'] = [df['Age'].mean()]
T['sum'] = [df['Age'].sum()]
T['cnt'] = [df['Age'].count()]
statistics = T.to_markdown()

# Unique values
T = pd.DataFrame()
T['unique_sex'] = df['Sex'].unique()
T['value_counts'] = [df['Sex'].value_counts()[0], df['Sex'].value_counts()[1]] 
unique_values = T.to_markdown()

# Value counts
T = pd.DataFrame()
T['PClass_value_counts'] = df['PClass'].value_counts()
value_counts = T.to_markdown()

# Missing values
df = df[df['Age'].isnull()]
missing_values = df.head(2).to_markdown()

print("Statistics:"); print(statistics, "\n")
print("Unique values:"); print(unique_values, "\n")
print("Value counts:"); print(value_counts, "\n")
print('Missing values | Age Null:'); print(missing_values, "\n")

"""
Statistics:
|    |   max |   min |    avg |     sum |   cnt |
|---:|------:|------:|-------:|--------:|------:|
|  0 |    71 |  0.17 | 30.398 | 22980.9 |   756 | 

Unique values:
|    | unique_sex   |   value_counts |
|---:|:-------------|---------------:|
|  0 | female       |            851 |
|  1 | male         |            462 | 

Value counts:
|     |   PClass_value_counts |
|:----|----------------------:|
| 3rd |                   711 |
| 1st |                   322 |
| 2nd |                   279 |
| *   |                     1 | 

Missing values | Age Null:
|    | Name                         | PClass   |   Age | Sex    |   Survived |   SexCode |
|---:|:-----------------------------|:---------|------:|:-------|-----------:|----------:|
| 12 | Aubert, Mrs Leontine Pauline | 1st      |   nan | female |          1 |         1 |
| 13 | Barkworth, Mr Algernon H     | 1st      |   nan | male   |          1 |         0 |
"""