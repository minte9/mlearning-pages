""" Read CSV
The source can be URL or FILE
"""

import pandas as pd
import pathlib

# Read from URL
URL = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/data.csv'
df1 = pd.read_csv(URL)

# Read from FILE
DIR = pathlib.Path(__file__).resolve().parent 
df2 = pd.read_csv(DIR / '../_data/01.csv')

print(df1.head(2).to_markdown(), "\n")
print(df2.head(2).to_markdown())

"""
    |    |   integer | datetime            |   category |
    |---:|----------:|:--------------------|-----------:|
    |  0 |         5 | 2015-01-01 00:00:00 |          0 |
    |  1 |         5 | 2015-01-01 00:00:01 |          0 | 

    |    |   integer | datetime            |   category |
    |---:|----------:|:--------------------|-----------:|
    |  0 |         5 | 2015-01-01 00:00:00 |          0 |
    |  1 |         5 | 2015-01-01 00:00:01 |          0 |
"""