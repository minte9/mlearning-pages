""" Decision Trees / Info Gain
Target: Play Tennis or not

Information gain example (for feature 'wind'):
IG = H - (8/14)H_weak - (6/14)H_strong
IG = 0.940 - (8/14)0.811 - (6/14)1.00 = 0.048 

Machine epsilon is the upper bound on the relative error due to rounding.
This small value added to the denominator in order to avoid division by zero. 
"""

import numpy as np
import pandas as pd
import pathlib

# Load the dataset from a CSV file
DIR = pathlib.Path(__file__).resolve().parent
df = pd.read_csv(DIR / 'data/play_tennis.csv')

# Split the dataset into features (X) and target (y), play tennis yes/no
X = df.drop(['play'], axis=1)
y = df['play'] 


""" Calculate the total entropy of the dataset 
"""
total_entropy = 0

# For each unique play outcome (yes/no)
for target in df['play'].unique():

    # Probability of target outcome
    target_count = df['play'].value_counts()[target]
    target_total = len(df['play'])
    P = target_count/target_total

    # Update total entropy
    total_entropy += -P*np.log2(P)


""" Function to calculate entropy of a specific attribute
"""
def attribute_entropy(attr):
    E = 0
    eps = np.finfo(float).eps  # Calculate machine epsilon for float operations

    targets = df.play.unique() # Unique target outcomes (yes/no)
    values = df[attr].unique() # Unique values for the given attribute (cool,hot)

    # For each unique value of the attribute 
    for v in values:
        # Initialize entropy for this value
        ent = 0

        # For each unique play outcome (yes/no)
        for t in targets:

            # Count occurrences where attribute=value and play outcome matches
            numerator = len(df[attr][df[attr] == v][df.play == t]) # numerator
            # Count occurrences where attribute=value
            denominator = len(df[attr][df[attr] == v])

            # Calculate probability
            P = numerator/(denominator + eps)
            # Update entropy for this value
            ent += -P*np.log2(P + eps)

        # Update total entropy using the weighted entropy for this value
        E += -(denominator/len(df))*ent

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
    IG[k] = total_entropy - E[k]

# Asserts
assert E['outlook']  < E['humidity']
assert IG['outlook'] > IG['humidity']

# Output results
print("\n Dataset:"); print(df)
print("\n Describe:"); print(df.describe())
print("\n Entropy:"); print(total_entropy)
print("\n AttrEntropy:"); print(E)
print("\n Information gains:"); print(IG)

"""
    Dataset:
        outlook  temp humidity  windy play
    0      sunny   hot     high  False   no
    1      sunny   hot     high   True   no
    2   overcast   hot     high  False  yes
    3      rainy  mild     high  False  yes
    4      rainy  cool   normal  False  yes
    5      rainy  cool   normal   True   no
    6   overcast  cool   normal   True  yes
    7      sunny  mild     high  False   no
    8      sunny  cool   normal  False  yes
    9      rainy  mild   normal  False  yes
    10     sunny  mild   normal   True  yes
    11  overcast  mild     high   True  yes
    12  overcast   hot   normal  False  yes
    13     rainy  mild     high   True   no

    Describe:
        outlook  temp humidity  windy play
    count       14    14       14     14   14
    unique       3     3        2      2    2
    top      sunny  mild     high  False  yes
    freq         5     6        7      8    9

    Entropy:
    0.9402859586706311

    AttrEntropy:
    {'outlook': 0.6935361388961914, 
     'temp': 0.9110633930116756, 
     'humidity': 0.7884504573082889, 
     'windy': 0.892158928262361}

    Information gains:
    {'outlook': 0.24674981977443977, 
     'temp': 0.029222565658955535, 
     'humidity': 0.15183550136234225, 
     'windy': 0.048127030408270155}
"""