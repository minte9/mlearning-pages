""" Read csv
pp27
Import a comma-separated values (CSV) file

https://github.com/chrisalbon/sim_data/

    int             datetime      category
    5    2015-01-01 00:00:00             0
    5    2015-01-01 00:00:01             0
    ...
"""

import pandas as pd
import pathlib

URL = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/data.csv'
FILE = pathlib.Path(__file__).resolve().parent / 'data/01.csv'

dataframe = pd.read_csv(URL)
print(dataframe.head(2))

dataframe = pd.read_csv(FILE)
print(dataframe.head(2))