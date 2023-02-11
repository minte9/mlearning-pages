""" Read csv
pp27
Import a comma-separated values (CSV) file
The source can be URL of FILE
https://github.com/chrisalbon/sim_data/
"""

import pandas as pd
import pathlib

# Read from URL
URL = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/data.csv'
dataframe = pd.read_csv(URL)

# Read from FILE
DIR = pathlib.Path(__file__).resolve().parent 
dataframe = pd.read_csv(DIR / '../_data/01.csv')

print(dataframe.head(2).to_markdown())

# |    |   integer | datetime            |   category |
# |---:|----------:|:--------------------|-----------:|
# |  0 |         5 | 2015-01-01 00:00:00 |          0 |
# |  1 |         5 | 2015-01-01 00:00:01 |          0 |