""" Pandas / DataFrame
"""

import pandas as pd

data = {
    'apples': [3, 2, 0, 1],
    'oranges': [0, 3, 7, 2],
    'available': ['yes', 'no', 'yes', 'no'],
}
df = pd.DataFrame(data)

print("DataFrame:")
print(df.to_markdown())

"""
DataFrame:
|    |   apples |   oranges | available   |
|---:|---------:|----------:|:------------|
|  0 |        3 |         0 | yes         |
|  1 |        2 |         3 | no          |
|  2 |        0 |         7 | yes         |
|  3 |        1 |         2 | no          |
"""
