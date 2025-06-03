import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes

# Diabetes dataset
dataset = load_diabetes()
X = dataset.data
y = dataset.target
features = dataset.feature_names

# Describe dataset
print("Keys:", dataset.keys())
print("Shape:", dataset['data'].shape)
print("Target values examples:", dataset.target[0:7])
print(dataset['DESCR'])

"""
    Keys: dict_keys(['data', 'target', 'frame', 'DESCR', ...])
    Shape: (442, 10)
    Target values examples: [151.  75. 141. 206. 135.  97. 138.]
    Diabetes dataset
    ----------------
    Ten baseline variables, age, sex, body mass index, average blood
    pressure, and six blood serum measurements were obtained for each of n =
    442 diabetes patients, as well as the response of interest, a
    quantitative measure of disease progression one year after baseline.
"""