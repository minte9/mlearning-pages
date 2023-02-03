""" KNN Algorithm Diabetes 

Target is a quantitative measure of disease progression 
one year after baseline. The target values are continuous not categorical,
you need to divide the target values into two categories.

    1 - represents target value greater than the mean target value
    0 - represents target value less than or equal
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes


# ---------------------------------------------
# Dataset

dataset = load_diabetes()

print(dataset.keys())
    # 'data', 'target', 'frame', 'DESCR', 'feature_names', 
    # 'data_filename', 'target_filename', 'data_module'

print(dataset['DESCR'])
    # Ten baseline variables, age, sex, body mass index, average blood
    # pressure, and six blood serum measurements were obtained for each of 
    # n = 442 diabetes patients

print(dataset['data'].shape) 
    # 442, 10

print(dataset.target[0:7])
    # 151.  75. 141. 206. 135.  97. 138.

X = dataset.data
y = dataset.target
features = dataset.feature_names

# Select two features to plot (bmi, bp)
X = X[:, [2, 3]]

# Make the target categorical
y_binary = np.where(y > y.mean(), 1, 0)

# Array containing class names (0 is 'OK', 1 is 'NOK')
class_names = ['OK', 'NOK']


# ------------------------------------------------
# Knn algorithm

# Define the unknown points
xA = np.array([-0.075, 0.040])
xB = np.array([0.10, 0.035])

# Define the number of nearest neighbors (k=5)
k = 5

def knn(k, x_unknown):
    distances = np.sqrt(np.sum((X - x_unknown)**2, axis=1))
    keys = np.argsort(distances)
    knn_keys = keys[:k]
    knn_classes = y_binary[knn_keys]
    knn_classes = knn_classes.astype(int)
    knn_most_common_class = np.bincount(knn_classes).argmax()
    knn_class = class_names[knn_most_common_class]
    return knn_class, knn_keys

knn_classA, knn_keysA = knn(k, xA)
knn_classB, knn_keysB = knn(k, xB)

# -----------------------------------------------
# Plotting

fig, ax = plt.subplots()
ax.set_xlabel('bmi (body mass index)')
ax.set_ylabel('bp (average blood pressure)') 

plt.scatter(X[:, 0], X[:, 1], c=y_binary)
plt.scatter(xA[0], xA[1], marker='o', color='g', label=knn_classA)
plt.scatter(xB[0], xB[1], marker='x', color='r', label=knn_classB)

for i in knn_keysA:
    plt.plot((xA[0], X[i][0]), (xA[1], X[i][1]), 
        color='gray', linestyle='--')

for i in knn_keysB:
    plt.plot((xB[0], X[i][0]), (xB[1], X[i][1]), 
        color='gray', linestyle='--')

plt.title('Diabetes dataset - KNN')
plt.legend()
plt.show()