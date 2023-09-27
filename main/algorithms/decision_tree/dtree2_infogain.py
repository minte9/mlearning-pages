""" Decision Trees / Info Gain
Play Tennis

IG = H - (8/14)H_weak - (6/14)H_strong
IG = 0.940 - (8/14)0.811 - (6/14)1.00 - 0.048 

Machine epsilon is the upper bound on the relative error due to rounding.
This small value added to the denominator in order to avoid division by zero. 
"""

import numpy as np
import pandas as pd
import pathlib

# Load the dataset from a CSV file
DIR = pathlib.Path(__file__).resolve().parent
df = pd.read_csv(DIR / 'data/play_tennis.csv')

# Split the dataset into features (X) and the target (y), whether play tennis or not
X = df.drop(['play'], axis=1)
y = df['play'] 

# Function to calculate the total entropy of the dataset
def dataset_entropy():
    E = 0

    # Count the occurrences of each play outcome (yes=9, no=5)
    N = df['play'].value_counts()

    # For each unique play outcome (yes/no)
    for v in df['play'].unique():

        # Calculate the probability of this outcome
        P = N[v]/len(df['play'])

         # Update total entropy using the probability
        E += -P*np.log2(P)
    return E

# Function to calculate the entropy of a specific attribute
def attribute_entropy(attr):
    E = 0

    # Calculate machine epsilon for float operations
    eps = np.finfo(float).eps 

    # Get unique play outcomes (yes/no)
    targets = df.play.unique()

    # Get unique values for the given attribute
    values = df[attr].unique()

    # For each unique value of the attribute (cool/hot)
    for v in values:

        # Initialize entropy for this value
        ent = 0

        # For each unique play outcome (yes/no)
        for t in targets:

            # Count occurrences where attribute=value and play outcome matches
            num = len(df[attr][df[attr] == v][df.play == t]) # numerator

            # Count occurrences where attribute=value
            den = len(df[attr][df[attr] == v])

            # Calculate a fraction related to the probability
            fraction = num/(den + eps)

            # Update entropy for this value
            ent += -fraction*np.log2(fraction + eps)

        # Update total entropy using the weighted entropy for this value
        E += -(den/len(df))*ent # sum of all entropies

    # Return the absolute value of the entropy
    return abs(E)


# Get the names of attributes (excluding the target variable)
attributes = df.keys()[:-1]

# Calculate entropy for each attribute and store it in a dictionary
E = {} 
for k in attributes:
    E[k] = attribute_entropy(k)

# Calculate information gain for each attribute and store it in a dictionary
IG = {}
for k in E:
    IG[k] = dataset_entropy() - E[k]

# Alternatives one-liner versions to calculate entropy and information gain
E  = {k:attribute_entropy(k) for k in attributes}
IG = {k:(dataset_entropy() - E[k]) for k in E} 

# Asserts
assert E['outlook']  < E['humidity']
assert IG['outlook'] > IG['humidity'] # Look Here

# Output results
print("\n Dataset:"); print(df)
print("\n Describe:"); print(df.describe())
print("\n Entropy:"); print(dataset_entropy())
print("\n AttrEntropy:"); print(E)
print("\n Information gains:"); print(IG)