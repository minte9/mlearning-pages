""" Read Excel
pp28
Import an Excel spreadsheet
pip install openpyxl
"""

import pandas as pd
import pathlib

URL = 'https://raw.githubusercontent.com/chrisalbon/sim_data/master/data.xlsx'
FILE = pathlib.Path(__file__).resolve().parent / 'data/02.xlsx'

df = pd.read_excel(URL, sheet_name=0)
print(df.head(2))

df = pd.read_excel(FILE, sheet_name=0)
print(df.head(2))