""" Iris Species Classifier
Learn model that predicts the species of a new iris
based on known measurements (length and width of petals).

Dataset is a Bunch object (similar to a dictionary).
The shape of the data array is the number of samples
times the number of features.

The target array contains the species of the flowers.
The species are encoded as integers from 0 to 2.

We use a fixed seed when splitting in train and test data.
One of the best way to inspect the data is to visualize it.

The data points are colored according to the species the iris belons to.
The diagonal of matrix is filled with histograms of each feature.

From the plots, we can see that tha three classes are well separated.
This means that machine learning model will be able to learn to separate them.
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

dataset = load_iris()
print('Keys: ', dataset.keys())
    # 'data', 'target', 'frame', 'target_names', 
    # 'DESCR', 'feature_names', 'filename', 'data_module'

print(dataset['DESCR'][:193])
    # Number of Instances: 150 (50 in each of three classes)
    # Number of Attributes: 4 numeric

print(dataset['feature_names'])
    # sepal length (cm), sepal width (cm)
    # petal length (cm), petal width (cm)

print(dataset['data'].shape)
    # (150, 4)

print(dataset['data'][:2])
    # [5.1 3.5 1.4 0.2]
    # [4.9 3.  1.4 0.2]

print(dataset['target'][:2])
print(dataset['target'][148:])
    # [0, 0]
    # [2, 2]

X1, X2, y1, y2 = train_test_split(
    dataset['data'], dataset['target'], random_state=0
)

print('X1 shape: ', X1.shape) # (112, 4)
print('X2 shape: ', X2.shape) # (38, 4)
print('y1 shape: ', y1.shape) # 112
print('y2 shape: ', y2.shape) # 38

df = pd.DataFrame(X1, columns=dataset.feature_names)
pd.plotting.scatter_matrix(
    df, c=y1, figsize=(15, 15), marker='o', 
    hist_kwds={'bins': 20}, s=60, alpha = .8
)
plt.show()