""" Make classification - Simulated dataset
pp025
Create simulated data for classification.
"""

import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

features, target = make_classification(
    n_samples = 100,
    n_features = 2,
    n_informative = 2,
    n_redundant = 0,
    n_classes = 2,
    weights = [.25, .75],
    random_state = 1
)

print(features[0:3])
    # [[ 1.30022717 -0.7856539 ]
    #  [ 1.44184425 -0.56008554]
    #  [-0.84792445 -1.36621324]]

print(target[:10])
    # [0 1 1 1 2 2 2 1 0 0]

plt.scatter(features[:, 0], features[:, 1], c=target)
plt.title('Make classification - Simultated dataset')
plt.show()