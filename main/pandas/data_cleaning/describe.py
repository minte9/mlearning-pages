""" Describe data
Real world cases could have millions of rows and columns.
We rely on pulling samples and summary statistics.

Describe do not always tell the full story.
Survived is categorical, but pandas treats it as numerical. 
Both iloc and loc are very useful during data cleaning.

For output data (outside Jupyter) use DataFrame' to_markdown()
pip install tabulate
"""

import pandas as pd
import pathlib

DIR = pathlib.Path(__file__).resolve().parent / '../_data/'
df = pd.read_csv(DIR / 'titanic.csv')

print("Show dimensions | shape:")
print(df.shape) # (1313, 6)

print("First two rows | head(2): ")
print(df.head(2).to_markdown())

# |    | Name               | PClass   |   Age | Sex    |   Survived |   SexCode |
# |---:|:------------------ |:---------|------:|:-------|-----------:|----------:|
# |  0 | Allen, Miss Elisab | 1st      |    29 | female |          1 |         1 |
# |  1 | Allison, Miss Hele | 1st      |     2 | female |          0 |         1 |


print("Show statistics | describe():")
print(df.describe().to_markdown())

# |       |     Age |    Survived |     SexCode |
# |:------|--------:|------------:|------------:|
# | count | 756     | 1313        | 1313        |
# | mean  |  30.398 |    0.342727 |    0.351866 |
# | std   |  14.259 |    0.474802 |    0.477734 |
# | min   |   0.17  |    0        |    0        |
# | 25%   |  21     |    0        |    0        |
# | 50%   |  28     |    0        |    0        |
# | 75%   |  39     |    1        |    1        |
# | max   |  71     |    1        |    1        |



print("Select first row by index | iloc[0]:")
print(df.iloc[0].to_markdown()) # first

# |          | 0                            |
# |:---------|:-----------------------------|
# | Name     | Allen, Miss Elisabeth Walton |
# | PClass   | 1st                          |
# | Age      | 29.0                         |
# | Sex      | female                       |
# | Survived | 1                            |
# | SexCode  | 1                            |


print("Second, third and fourth | iloc[1:4]:")
print(df.iloc[1:4].to_markdown()) # second, third and fourth

print("Select up to and including fourth | iloc[:4]")
print(df.iloc[:4].to_markdown())  # up to, and including fourth


# Set index to non-numerical
df = df.set_index(df['Name'])
print("Select by Name:") 
print(df.loc['Allen, Miss Elisabeth Walton'].to_markdown())

# |          | Allen, Miss Elisabeth Walton   |
# |:---------|:-------------------------------|
# | Name     | Allen, Miss Elisabeth Walton   |
# | PClass   | 1st                            |
# | Age      | 29.0                           |
# | Sex      | female                         |
# | Survived | 1                              |
# | SexCode  | 1                              |