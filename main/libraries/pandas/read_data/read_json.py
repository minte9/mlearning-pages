""" Read data / Json
Load a JSON file for data preprocessing.
"""

import pandas as pd
import pathlib

# File path
DIR = pathlib.Path(__file__).resolve().parent 
FILE = DIR / '_data/03.json'

# Read from json
print("Load from json file:")
df = pd.read_json(
        FILE, orient='columns'
)
print(df.head(2).to_markdown())

# Read from string
print("Load from json string:")
data = [
    {"id": 1, "name": "Mary"},
    {"id": 2, "name": "John"},
]
df = pd.json_normalize(data)
print(df.head(2).to_markdown())

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