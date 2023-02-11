""" Read Json
pp29
Load a JSON file for data preprocessing.

http://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.json_normalize.html
"""

import pandas as pd
import pathlib

URL = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/data.json'
FILE = pathlib.Path(__file__).resolve().parent / 'data/03.json'

df = pd.read_json(URL, orient='columns')
print(df.head(2))

df = pd.read_json(FILE, orient='columns')
print(df.head(2))

# Using pandas json_normalize
DATA = [
    {
        "id": 1,
        "name": "Mary",
    },
    {
        "id": 2,
        "name": "John",
    },
]

df = pd.json_normalize(DATA)
print(df.head(2))
    #     id  name
    # 0   1   Mary
    # 1   2   John