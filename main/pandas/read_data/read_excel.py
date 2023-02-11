""" Read Excel
pp28
Import an Excel spreadsheet
pip install openpyxl
"""

import pandas as pd
import pathlib

FILE = pathlib.Path(__file__).resolve().parent / '../_data/02.xlsx'
df = pd.read_excel(FILE , sheet_name=0)

print(df.head(2).to_markdown())

# |    |   integer | datetime            |   category |
# |---:|----------:|:--------------------|-----------:|
# |  0 |         5 | 2015-01-01 00:00:00 |          0 |
# |  1 |         5 | 2015-01-01 00:00:01 |          0 |