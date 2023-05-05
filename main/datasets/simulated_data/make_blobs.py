""" Make blobs
Blobs are dataset that work well with clustering techniques.
"""

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Make blob
features, target = make_blobs(
    n_samples = 100,
    n_features = 2,
    centers = 3, # three target classes
    cluster_std = 0.5,
    shuffle = True,
    random_state = 1
)

plt.scatter(features[:, 0], features[:, 1], c=target)
plt.title('Make blob - Simultated dataset')
plt.show()

print("Features[0:3]:\n", features[0:3])
print("Target:[:10]:", target[:10])

"""
    Features[0:3]:
     [[ -1.22685609   3.25572052]
      [ -9.57463218  -4.38310652]
      [-10.71976941  -4.20558148]]
    Target:[:10]: [0 1 1 1 2 2 2 1 0 0]
"""