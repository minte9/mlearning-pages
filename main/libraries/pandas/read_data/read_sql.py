""" Read data / SQL Queries
"""

import pandas as pd
import sqlite3
import pathlib

# File path
DIR = pathlib.Path(__file__).resolve().parent
DB = DIR / '_data/04.db'

conn = sqlite3.connect(DB)
df = pd.read_sql_query("SELECT * FROM data", conn)

print("DB:")
print(df.head(2).to_markdown())

"""
DataFrame from DB:
|    | first_name   | last_name   |   age |   preTestScore |   postTestScore |
|---:|:-------------|:------------|------:|---------------:|----------------:|
|  0 | Jason        | Miller      |    42 |              4 |              25 |
|  1 | Molly        | Jacobson    |    52 |             24 |              94 |
"""
