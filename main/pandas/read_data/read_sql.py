""" Read data / SQL Database
Load data from a database using SQL queries.
Probably the most used in real world.
"""

import pandas as pd
import sqlite3
import pathlib

DIR = pathlib.Path(__file__).resolve().parent
DB = DIR / 'data/04.db'

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
