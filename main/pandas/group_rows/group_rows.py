""" Grouping rows by values

Groupby is one of the most powerful feature in pandas.
We can group rows according to some shared value.

We can also group by a first column, the group that grouping 
by a second column.
"""

import pandas as pd
import pathlib

DIR = pathlib.Path(__file__).resolve().parent / '../_data/'
df = pd.read_csv(DIR / 'titanic.csv')


print("Grouped by Sex averages:")
print(df.groupby('Sex').mean(numeric_only=True).to_markdown())

# | Sex    |     Age |   Survived |   SexCode |
# |:-------|--------:|-----------:|----------:|
# | female | 29.3964 |   0.666667 |         1 |
# | male   | 31.0143 |   0.166863 |         0 |


print("Grouped by Sex count:")
print(df.groupby('Sex')['Name'].count().to_markdown())

# | Sex    |   Name |
# |:-------|-------:|
# | female |    462 |
# | male   |    851 |


print("Grouped by Survived count:")
print(df.groupby('Survived')['Name'].count().to_markdown())

# |   Survived |   Name |
# |-----------:|-------:|
# |          0 |    863 |
# |          1 |    450 |


print("Group by two columns (average):")
print(df.groupby(['Sex', 'Survived'])['Age'].mean().to_markdown())

# |               |     Age |
# |:--------------|--------:|
# | ('female', 0) | 24.9014 |
# | ('female', 1) | 30.8671 |
# | ('male', 0)   | 32.3208 |
# | ('male', 1)   | 25.9519 |