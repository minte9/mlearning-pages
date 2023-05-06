""" Read data / Excel

Import an Excel spreadsheet
    pip install openpyxl
"""

import pandas as pd
import pathlib

DIR = pathlib.Path(__file__).resolve().parent 
FILE = DIR / '../_data/02.xlsx'
df = pd.read_excel(FILE , sheet_name=0)

print("Read csv from excel:")
print(df.head(2).to_markdown())

"""
    Read csv from excel:
    |    |   integer | datetime            |   category |
    |---:|----------:|:--------------------|-----------:|
    |  0 |         5 | 2015-01-01 00:00:00 |          0 |
    |  1 |         5 | 2015-01-01 00:00:01 |          0 |
"""