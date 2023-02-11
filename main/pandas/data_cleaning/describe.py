""" Describe data
pp35
https://github.com/chrisalbon/sim_data/

Real world cases could have millions of rows and columns.
We rely on pulling samples and summary statistics.

Describe do not always tell the full story.
Survived is categorical, but pandas treats it as numerical. 

Both iloc and loc are very useful during data cleaning.

            Name    PClass   Age     Sex  Survived  SexCode
Allen, Elisabeth    1st     29.0  female         1        1
Allison, Helen      1st      2.0  female         0        1

              Age     Survived      SexCode
count  756.000000  1313.000000  1313.000000
mean    30.397989     0.342727     0.351866
std     14.259049     0.474802     0.477734
min      0.170000     0.000000     0.000000
25%     21.000000     0.000000     0.000000
50%     28.000000     0.000000     0.000000
75%     39.000000     1.000000     1.000000
max     71.000000     1.000000     1.000000
"""

import pandas as pd
import pathlib

DIR = pathlib.Path(__file__).resolve().parent / '../data/'
df = pd.read_csv(DIR / 'titanic.csv')

# Show two rows
print(df.head(2))
print()

# Show dimensions
print(df.shape) # (1313, 6)
print()

# Show statistics
print(df.describe())
print()

# Select rows
print(df.iloc[0])   # first
print(df.iloc[1:4]) # second, third and fourth
print(df.iloc[:4])  # up to, and including fourth
print()

# Set index to non-numerical
df = df.set_index(df['Name'])
print(df.loc['Allen, Miss Elisabeth Walton'])
print()