""" Decision Trees / Prunning

Insteed of looking at the whole tree, we can select only the most useful properties.
To summarize the feature importance we use `tree.feature_importance`.
We can see that `worst radius` used in the top split, is by far 
the most important feature.
"""

import numpy as np
import pathlib
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Dataset
DIR = pathlib.Path(__file__).resolve().parent
df = load_breast_cancer()

# Training and test data
X1, X2, y1, y2 = train_test_split(
    df.data, df.target, stratify=df.target, random_state=42)

# Pre-prunning
dtree = DecisionTreeClassifier(max_depth=4, random_state=0)
dtree.fit(X1, y1)

# Predictions
X_new = X2[15]
y_pred = dtree.predict(X_new.reshape(1, -1))
y_pred_target = df['target_names'][y_pred]
score = dtree.score(X2, dtree.predict(X2))

# Output
n = df.data.shape[1]
plt.subplots_adjust(left=0.28)
plt.barh(np.arange(n), dtree.feature_importances_, align='center')
plt.yticks(np.arange(n), df.feature_names)
plt.xlabel("Feature importance")
plt.ylabel("Feature")
plt.ylim(-1, n)
# plt.show()

outputs = [
    ["Featre names:", df['feature_names']],
    ["Dataset:", df['data']],
    ["Shape:", df['data'].shape],
    ["X_new:", X_new],
    ["Prediction:", y_pred],
    ["Prediction Target:", y_pred_target],
    ["Score:", score],
    ['Feature importances:', dtree.feature_importances_],
]
for out in outputs:
    print("\n", out[0], "\n ", out[1])


"""
	 Featre names: 
	  [ 'mean radius' 'mean texture' 'mean perimeter' 'mean area'
        'mean smoothness' 'mean compactness' 'mean concavity'
        'mean concave points' 'mean symmetry' 'mean fractal dimension'
        'radius error' 'texture error' 'perimeter error' 'area error'
        'smoothness error' 'compactness error' 'concavity error'
        'concave points error' 'symmetry error' 'fractal dimension error'
        'worst radius' 'worst texture' 'worst perimeter' 'worst area'
        'worst smoothness' 'worst compactness' 'worst concavity'
        'worst concave points' 'worst symmetry' 'worst fractal dimension' ]

	 Dataset: 
	  [ [1.799e+01 1.038e+01 1.228e+02 ... 2.654e-01 4.601e-01 1.189e-01]
        [2.057e+01 1.777e+01 1.329e+02 ... 1.860e-01 2.750e-01 8.902e-02]
        [1.969e+01 2.125e+01 1.300e+02 ... 2.430e-01 3.613e-01 8.758e-02]
        ...
        [1.660e+01 2.808e+01 1.083e+02 ... 1.418e-01 2.218e-01 7.820e-02]
        [2.060e+01 2.933e+01 1.401e+02 ... 2.650e-01 4.087e-01 1.240e-01]
        [7.760e+00 2.454e+01 4.792e+01 ... 0.000e+00 2.871e-01 7.039e-02] ]

	 Shape: 
	  (569, 30)

	 X_new: 
	  [ 9.683e+00 1.934e+01 6.105e+01 2.857e+02 8.491e-02 5.030e-02 2.337e-02
        9.615e-03 1.580e-01 6.235e-02 2.957e-01 1.363e+00 2.054e+00 1.824e+01
        7.440e-03 1.123e-02 2.337e-02 9.615e-03 2.203e-02 4.154e-03 1.093e+01
        2.559e+01 6.910e+01 3.642e+02 1.199e-01 9.546e-02 9.350e-02 3.846e-02
        2.552e-01 7.920e-02 ]

	 Prediction: 
	  [1]
     
     Prediction Target: 
     ['benign']

	 Score: 
	  1.0

	 Feature importances: 
	  [0.         0.         0.         0.         0.         0.
	 0.         0.         0.         0.         0.01019737 0.04839825
	 0.         0.         0.0024156  0.         0.         0.
	 0.         0.         0.72682851 0.0458159  0.         0.
	 0.0141577  0.         0.018188   0.1221132  0.01188548 0.        ]
"""
