import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes

# Diabetes dataset
dataset = load_diabetes()
X = dataset.data
y = dataset.target
features = dataset.feature_names

# Select 2 features to plot (bmi, bp) and make the target categorical
X = X[:, [2, 3]]                        
y_binary = np.where(y > y.mean(), 1, 0)
class_names = ['OK', 'NOK']  # 0 is 'OK', 1 is 'NOK'

# KNN Algorithm
def knn(x_unknown, k=5):
    distances = np.sqrt(np.sum((X - x_unknown)**2, axis=1))

    keys = np.argsort(distances)
    knn_keys = keys[:k]

    knn_classes = y_binary[knn_keys]
    knn_classes = knn_classes.astype(int)

    knn_most_common_class = np.bincount(knn_classes).argmax()
    knn_class = class_names[knn_most_common_class]
    return knn_class, knn_keys

# Unknown points
xA = np.array([-0.075, 0.040])
xB = np.array([0.10, 0.035])
k_nearest = 5 # number of nearest neighbors (k=5)

# Predictions
knn_classA, knn_keysA = knn(xA, k_nearest)
knn_classB, knn_keysB = knn(xB, k_nearest)


# Plot the two unknown points and dataset points
fig, ax = plt.subplots()
ax.set_xlabel('bmi (body mass index)')
ax.set_ylabel('bp (average blood pressure)') 

plt.scatter(X[:, 0], X[:, 1], c=y_binary)
plt.scatter(xA[0], xA[1], marker='o', color='g', label=knn_classA)
plt.scatter(xB[0], xB[1], marker='x', color='r', label=knn_classB)

# Plot lines to the nearest points
for i in knn_keysA:
    plt.plot((xA[0], X[i][0]), (xA[1], X[i][1]), color='gray', linestyle='--')
for i in knn_keysB:
    plt.plot((xB[0], X[i][0]), (xB[1], X[i][1]), color='gray', linestyle='--')

plt.title('Diabetes dataset - Knn')
plt.legend()
plt.show()