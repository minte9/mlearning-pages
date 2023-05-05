""" Iris Species  / Dataset description

It's difficult to plot datasets with more than 2-3 features.
Pair plots uses all posible pair of features.  

The data points are colored according to the species the iris belons to.
From the plots, we can see that tha three classes are well separated.
This means that ML model will be able to learn to separate them.
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

# Dataset
dataset = load_iris()

# Training and test data
X1, X2, y1, y2 = train_test_split(
    dataset['data'], dataset['target'], random_state=0 # fixed seed
)

# Plot features matrix
df = pd.DataFrame(X1, columns=dataset.feature_names)
pd.plotting.scatter_matrix(
    df, c=y1, figsize=(15, 15), marker='o', 
    s=60, alpha = .8, diagonal='none'
)
plt.suptitle('Iris features matrix')
plt.show()

# Describe dataset  
print('Description: \n ', dataset['DESCR'][:193])
print('Keys: ', dataset.keys())
print('Target names: ', dataset['target_names'])
print('Feature_names: ', dataset['feature_names'])
print('Shape: ', dataset['data'].shape) # number of samples, features
print('Data[:2]:\n ', dataset['data'][:2])
print('Target[:2]: ', dataset['target'][:2])
print('Target[148:]: ', dataset['target'][148:]) # species are encoded, 0 to 2
print('X1 shape: ', X1.shape)
print('X2 shape: ', X2.shape)
print('y1 shape: ', y1.shape)
print('y2 shape: ', y2.shape)

"""
    Keys: dict_keys(['data', 'target', 'frame', 'target_names', ... module'])
    Target names:  ['setosa' 'versicolor' 'virginica']
    Feature_names:  ['sepal length (cm)', 'sepal width (cm)', ...]
    Shape:  (150, 4)
    Data[:2]:
    [[5.1 3.5 1.4 0.2]
    [4.9 3.  1.4 0.2]]
    Target[:2]:  [0 0]
    Target[148:]:  [2 2]
    X1 shape:  (112, 4)
    X2 shape:  (38, 4)
    y1 shape:  (112,)
    y2 shape:  (38,)
"""