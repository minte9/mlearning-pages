""" Blobs
Blobs are dataset that work well with clustering techniques.
"""

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.datasets import make_classification

# Make blob
features1, target1 = make_blobs(
    n_samples = 100,
    n_features = 2,
    centers = 3, # three target classes
    cluster_std = 0.5,
    shuffle = True,
    random_state = 1
)

# Make classification
features2, target2 = make_classification(
    n_samples = 100,
    n_features = 2,
    n_informative = 2,
    n_redundant = 0,
    n_classes = 2,
    weights = [.25, .75],
    random_state = 1
)

# Plot blobs
plt.scatter(features1[:, 0], features1[:, 1], c=target1)
plt.title('Make blob - Simultated dataset')
plt.scatter(features2[:, 0], features2[:, 1], c=target2)
plt.title('Make classification - Simultated dataset')
plt.show()

print("Blob Features[0:3]:\n", features[0:3])
print("Blob Target:[:10]:", target[:10])
print("Classification Features[0:3]:\n", features[0:3])
print("Classification Target:[:10]:", target[:10])

"""
    Features[0:3]:
     [[ -1.22685609   3.25572052]
      [ -9.57463218  -4.38310652]
      [-10.71976941  -4.20558148]]
    Target:[:10]: [0 1 1 1 2 2 2 1 0 0]
"""