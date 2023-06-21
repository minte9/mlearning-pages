""" Read data / CSV
The source can be URL or FILE
"""

import pandas as pd
import pathlib

URL = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/data.csv'
DIR = pathlib.Path(__file__).resolve().parent 
FILE = DIR / 'data/01.csv'

df1 = pd.read_csv(URL)
df2 = pd.read_csv(FILE)

print("URL:")
print(df1.head(2).to_markdown())

print("FILE:")
print(df2.head(2).to_markdown())

"""
Read csv from url:
|    |   integer | datetime            |   category |
|---:|----------:|:--------------------|-----------:|
|  0 |         5 | 2015-01-01 00:00:00 |          0 |
|  1 |         5 | 2015-01-01 00:00:01 |          0 |

Read csv from file:
|    |   integer | datetime            |   category |
|---:|----------:|:--------------------|-----------:|
|  0 |         5 | 2015-01-01 00:00:00 |          0 |
|  1 |         5 | 2015-01-01 00:00:01 |          0 |
"""