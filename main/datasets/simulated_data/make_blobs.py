""" The make_* functions in the sklearn.datasets module are used 
to generate synthetic datasets for machine learning tasks.

Make blobs generates isotropic Gaussian blobs for clustering tasks. 
Make classification generates a random n-class classification problem.
Make regression generates a random regression problem.
"""

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.datasets import make_classification
from sklearn.datasets import make_regression

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

# Make regression
features3, target3, coef3 = make_regression(
    n_samples = 100,
    n_features = 3,
    n_informative = 3,
    n_targets = 1,
    noise = 0,
    coef = True,
    random_state = 1
)

# Plot blobs
plt.scatter(features1[:, 0], features1[:, 1], c=target1)
plt.title('Make blob - Simultated dataset')
plt.show()

plt.scatter(features2[:, 0], features2[:, 1], c=target2)
plt.title('Make classification - Simultated dataset')
plt.show()

plt.scatter(features3[:, 0], features3[:, 1], c=target3)
plt.title('Make regression - Simultated dataset')
plt.show()

print("Blob / Features[0:3]:\n", features1[0:3])
print("Target[:10]:", target1[:10], '\n')

print("Classification / Features[0:3]:\n", features2[0:3])
print("Target[:10]:", target2[:10], '\n')

print("Regression / Features[0:3]:\n", features3[0:3])
print("Target[:10]:\n", target3[:10], '\n')

"""
	Blob / Features[0:3]:
	 [[ -1.22685609   3.25572052]
	  [ -9.57463218  -4.38310652]
	  [-10.71976941  -4.20558148]]
	Target:[:10]: [0 1 1 1 2 2 2 1 0 0] 

	Classification / Features[0:3]:
	 [[ 1.30022717 -0.7856539 ]
	  [ 1.44184425 -0.56008554]
	  [-0.84792445 -1.36621324]]
	Target:[:10]: [1 1 0 0 0 1 1 1 0 1]

	Regression / Features[0:3]:
	 [[ 1.29322588 -0.61736206 -0.11044703]
	  [-2.793085    0.36633201  1.93752881]
	  [ 0.80186103 -0.18656977  0.0465673 ]]
	Target[:10]:
	  [ -10.37865986   25.5124503    19.67705609  149.50205427 -121.65210879
	     90.29412996  214.01379719  224.74157328  -73.17331138 -195.62776209] 
"""