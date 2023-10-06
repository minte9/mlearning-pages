""" Statistics / Max, Min, Average
Can be applied to a column or to whole dataframe.
"""

import pandas as pd
import pathlib

DIR = pathlib.Path(__file__).resolve().parent
df = pd.read_csv(DIR / '../_data/titanic.csv')

# Statistics (by Age)
A = pd.DataFrame()
A['max'] = [df['Age'].max()]
A['min'] = [df['Age'].min()]
A['avg'] = [df['Age'].mean()]
print(A)

# Value counts (by PClass)
A = pd.DataFrame()
A['PClass'] = df['PClass'].value_counts()
print(A)

# Unique values (by Sex)
A = pd.DataFrame()
A['unique_value'] = df['Sex'].unique()
A['total'] = [df['Sex'].value_counts()[0], df['Sex'].value_counts()[1]]
print(A)

# Missing values (by Agge)
A = pd.DataFrame()
A = df[df['Age'].isnull()]

print("Missing values (Age):", A.size)
print(A.head())

"""

    max   min        avg
0  71.0  0.17  30.397989

     PClass
3rd     711
1st     322
2nd     279
*         1

  unique_value  total
0       female    851
1         male    462

Missing values (Age): 3342
                            Name PClass  Age     Sex  Survived  SexCode
12  Aubert, Mrs Leontine Pauline    1st  NaN  female         1        1
13      Barkworth, Mr Algernon H    1st  NaN    male         1        0
14            Baumann, Mr John D    1st  NaN    male         0        0
29       Borebank, Mr John James    1st  NaN    male         0        0
32            Bradley, Mr George    1st  NaN    male         1        0

"""