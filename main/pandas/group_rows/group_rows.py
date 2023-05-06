""" Grouping rows by values

Groupby is one of the most powerful feature in pandas.
We can group rows according to some shared value.

We can also group by a first column, the group that grouping 
by a second column.
"""

import pandas as pd
import pathlib

DIR = pathlib.Path(__file__).resolve().parent / '../_data/'
FILE = DIR / 'titanic.csv'
df = pd.read_csv(FILE)

T1 = df.groupby('Sex').count().to_markdown()
T2 = df.groupby('Sex').mean(numeric_only=True).to_markdown()
T3 = df.groupby('Sex')['Survived'].count().to_markdown()
T4 = df.groupby(['Sex', 'Survived'])['Age'].mean().to_markdown()

print("Grouped by Sex:"); print(T1, "\n")
print("Grouped by Sex averages:"); print(T2, "\n")
print("Grouped by Sex (count only Survived):"); print(T3, "\n")
print("Group by two columns (average):"); print(T4, "\n")

"""
Grouped by Sex:
| Sex    |   Name |   PClass |   Age |   Survived |   SexCode |
|:-------|-------:|---------:|------:|-----------:|----------:|
| female |    462 |      462 |   288 |        462 |       462 |
| male   |    851 |      851 |   468 |        851 |       851 | 

Grouped by Sex averages:
| Sex    |     Age |   Survived |   SexCode |
|:-------|--------:|-----------:|----------:|
| female | 29.3964 |   0.666667 |         1 |
| male   | 31.0143 |   0.166863 |         0 | 

Grouped by Sex (count only Survived):
| Sex    |   Survived |
|:-------|-----------:|
| female |        462 |
| male   |        851 | 

Group by two columns (average):
|               |     Age |
|:--------------|--------:|
| ('female', 0) | 24.9014 |
| ('female', 1) | 30.8671 |
| ('male', 0)   | 32.3208 |
| ('male', 1)   | 25.9519 | 
"""