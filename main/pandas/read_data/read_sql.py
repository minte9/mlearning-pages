""" Read SQL Database
Load data from a database using SQL queries.
Probably the most used in real world.
"""

import pandas as pd
import sqlite3
import pathlib

DIR = pathlib.Path(__file__).resolve().parent

conn = sqlite3.connect(DIR / '../_data/04.db')
df = pd.read_sql_query("SELECT * FROM data", conn)

print(df.head(2).to_markdown())

# |    | first_name   | last_name   |   age |   preTestScore |   postTestScore |
# |---:|:-------------|:------------|------:|---------------:|----------------:|
# |  0 | Jason        | Miller      |    42 |              4 |              25 |
# |  1 | Molly        | Jacobson    |    52 |             24 |              94 |
