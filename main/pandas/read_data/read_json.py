""" Read data / Json
Load a JSON file for data preprocessing.
"""

import pandas as pd
import pathlib

# Json file
DIR = pathlib.Path(__file__).resolve().parent 
FILE = DIR / '../_data/03.json'
df1 = pd.read_json(FILE, orient='columns')

# Json string
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
df2 = pd.json_normalize(DATA)

print("DataFrame from json file:")
print(df1.head(2).to_markdown(), "\n")

print("DataFrame from json string:")
print(df2.head(2).to_markdown())

"""
DataFrame from json file:
|    |   integer | datetime            |   category |
|---:|----------:|:--------------------|-----------:|
|  0 |         5 | 2015-01-01 00:00:00 |          0 |
|  1 |         5 | 2015-01-01 00:00:01 |          0 |

DataFrame from json string:
|    |   id | name   |
|---:|-----:|:-------|
|  0 |    1 | Mary   |
|  1 |    2 | John   |
"""