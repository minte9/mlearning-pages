""" Data clearing / Describe DataFrame

Real world cases could have millions of rows and columns.
Describe do not always tell the full story.

For example in `titanic.csv`, `survived` is categorical, 
but pandas treats it as numerical. 

Both iloc and loc are very useful during data cleaning.
For output data (outside Jupyter) use DataFrame' to_markdown()
    pip install tabulate
"""

import pandas as pd
import pathlib

DIR = pathlib.Path(__file__).resolve().parent / '../_data/'
FILE = DIR / 'titanic.csv'

# Read from csv
df = pd.read_csv(FILE)

# Set index to non-numerical
df2 = df.set_index(df['Name'])

first_row        = df.iloc[0]
second_to_fourth = df.iloc[1:4]
up_to_fourth     = df.iloc[:4]
by_name          = df2.loc['Allen, Miss Elisabeth Walton']

print("Shape:", df.shape, "\n") # (1313, 6)
print("First two rows | head(2): "); print(df.head(2).to_markdown(), "\n")
print("Show statistics | describe(): "); print(df.describe().to_markdown(), "\n")
print("Select first row by index | iloc[0]: "); print(first_row.to_markdown(), "\n")
print("Second, third and fourth | iloc[1:4]: "); print(second_to_fourth.to_markdown(), "\n")
print("Up to and including fourth | iloc[:4]: "); print(up_to_fourth.to_markdown(), "\n")
print("Select by Name: "); print(by_name.to_markdown(), "\n")

"""
Shape: (1313, 6)

First two rows | head(2): 
|    | Name                         | PClass   |   Age | Sex    |   Survived |   SexCode |
|---:|:-----------------------------|:---------|------:|:-------|-----------:|----------:|
|  0 | Allen, Miss Elisabeth Walton | 1st      |    29 | female |          1 |         1 |
|  1 | Allison, Miss Helen Loraine  | 1st      |     2 | female |          0 |         1 |

Show statistics | describe():
|       |     Age |    Survived |     SexCode |
|:------|--------:|------------:|------------:|
| count | 756     | 1313        | 1313        |
| mean  |  30.398 |    0.342727 |    0.351866 |
| std   |  14.259 |    0.474802 |    0.477734 |
| min   |   0.17  |    0        |    0        |
| 25%   |  21     |    0        |    0        |
| 50%   |  28     |    0        |    0        |
| 75%   |  39     |    1        |    1        |
| max   |  71     |    1        |    1        |

Select first row by index | iloc[0]:
|          | 0                            |
|:---------|:-----------------------------|
| Name     | Allen, Miss Elisabeth Walton |
| PClass   | 1st                          |
| Age      | 29.0                         |
| Sex      | female                       |
| Survived | 1                            |
| SexCode  | 1                            |

Second, third and fourth | iloc[1:4]:
|    | Name                                          | PClass   |   Age | Sex    |   Survived |   SexCode |
|---:|:----------------------------------------------|:---------|------:|:-------|-----------:|----------:|
|  1 | Allison, Miss Helen Loraine                   | 1st      |     2 | female |          0 |         1 |
|  2 | Allison, Mr Hudson Joshua Creighton           | 1st      |    30 | male   |          0 |         0 |
|  3 | Allison, Mrs Hudson JC (Bessie Waldo Daniels) | 1st      |    25 | female |          0 |         1 |

Select up to and including fourth | iloc[:4]
|    | Name                                          | PClass   |   Age | Sex    |   Survived |   SexCode |
|---:|:----------------------------------------------|:---------|------:|:-------|-----------:|----------:|
|  0 | Allen, Miss Elisabeth Walton                  | 1st      |    29 | female |          1 |         1 |
|  1 | Allison, Miss Helen Loraine                   | 1st      |     2 | female |          0 |         1 |
|  2 | Allison, Mr Hudson Joshua Creighton           | 1st      |    30 | male   |          0 |         0 |
|  3 | Allison, Mrs Hudson JC (Bessie Waldo Daniels) | 1st      |    25 | female |          0 |         1 |

Select by Name:
|          | Allen, Miss Elisabeth Walton   |
|:---------|:-------------------------------|
| Name     | Allen, Miss Elisabeth Walton   |
| PClass   | 1st                            |
| Age      | 29.0                           |
| Sex      | female                         |
| Survived | 1                              |
| SexCode  | 1                              |
"""