""" Breast cancer Dataset
"""

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
import pandas as pd

dataset = load_breast_cancer()
print(dataset['feature_names']) # 'mean radius' 'mean texture' ...
print(dataset['data'])          # [[1.799e+01 1.038e+01 1.228e+02 ... 
print(dataset['data'].shape)    # (569, 30)

X_train, X_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target, stratify=dataset.target, random_state=66)

training_accuracy = []
test_accuracy = []
neighbors = range(1, 11)

for n in neighbors:
    clf = KNeighborsClassifier(n_neighbors=n)
    clf.fit(X_train, y_train)
    training_accuracy.append(clf.score(X_train, y_train))
    test_accuracy.append(clf.score(X_test, y_test))

# optim n_neighbors
k, max_accuracy = max(enumerate(test_accuracy), key=lambda x: x[1])
print(k, max_accuracy)
n = neighbors[k]
print(n)

plt.plot(neighbors, training_accuracy, label="training accuracy")
plt.plot(neighbors, test_accuracy, label="test accuracy")
plt.plot([n, n], [0.88, 1], linestyle='--', label="optim n_neighbors")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
# plt.show()

# Cancer mean scatter plot
model = KNeighborsClassifier(n_neighbors=n)
model.fit(X_train, y_train)

X_unknown = X_test[15]
y_unknown = model.predict(X_unknown.reshape(1, -1))
print(X_unknown)
print(y_unknown)
y_target = dataset['target_names'][y_unknown]
print("Predicted target:", y_target)

fig, ax = plt.subplots()
ax.set_title("Breast Cancer")
ax.set_xlabel('mean area')
ax.set_ylabel('mean concavity')
df = pd.DataFrame(X_train, columns=dataset.feature_names)
ax.scatter(df['mean area'], df['mean concavity'], c=y_train)
ax.scatter(X_unknown[3], X_unknown[6], c='r',marker='x',s=100, label=y_target[0])
ax.grid()
plt.legend(loc='upper right')
plt.show()