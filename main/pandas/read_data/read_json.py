""" Read Json
pp29
Load a JSON file for data preprocessing.

http://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.json_normalize.html
"""

import pandas as pd
import pathlib

FILE = pathlib.Path(__file__).resolve().parent / '../_data/03.json'
df = pd.read_json(FILE, orient='columns')

print(df.head(2).to_markdown())

# |    |   integer | datetime            |   category |
# |---:|----------:|:--------------------|-----------:|
# |  0 |         5 | 2015-01-01 00:00:00 |          0 |
# |  1 |         5 | 2015-01-01 00:00:01 |          0 |


# Json normalize
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

print(df.head(2).to_markdown())

# |    |   id | name   |
# |---:|-----:|:-------|
# |  0 |    1 | Mary   |
# |  1 |    2 | John   |