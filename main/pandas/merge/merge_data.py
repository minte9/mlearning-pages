""" Merging DataFrames

In real world we usually are faced with multiple sources.
Sometimes we need to get all that data in one place.
For outer join, the 'how' parameter is used.
"""

import pandas as pd

employes = pd.DataFrame()
employes['id_employee'] = [1, 2, 3, 4]
employes['name'] = ['John', 'Mary', 'Bob', 'Michael']

sales = pd.DataFrame()
sales['id_employee'] = [3, 4, 5, 6]
sales['total_sales'] = [10000, 20000, 30000, 40000]

D1 = pd.merge(employes, sales, on='id_employee')
D2 = pd.merge(employes, sales, on='id_employee', how='outer')
D3 = pd.merge(employes, sales, left_on='id_employee', right_on='id_employee')

print("Inner join (default):"); print(D1.to_markdown(), "\n")
print("Outer join (how):"); print(D2.to_markdown(), "\n")
print("Column name (in each) to merge on:"); print(D3.to_markdown(), "\n")

"""
Inner join (default):
|    |   id_employee | name    |   total_sales |
|---:|--------------:|:--------|--------------:|
|  0 |             3 | Bob     |         10000 |
|  1 |             4 | Michael |         20000 | 

Outer join (how):
|    |   id_employee | name    |   total_sales |
|---:|--------------:|:--------|--------------:|
|  0 |             1 | John    |           nan |
|  1 |             2 | Mary    |           nan |
|  2 |             3 | Bob     |         10000 |
|  3 |             4 | Michael |         20000 |
|  4 |             5 | nan     |         30000 |
|  5 |             6 | nan     |         40000 | 

Column name (in each) to merge on:
|    |   id_employee | name    |   total_sales |
|---:|--------------:|:--------|--------------:|
|  0 |             3 | Bob     |         10000 |
|  1 |             4 | Michael |         20000 | 
"""