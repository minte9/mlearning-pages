""" Decision Trees / ID3 Algorithm

Iterative Dichotomiser 3, is a classification algorithm that follows a 
greedy approach of building a decision tree that gives priority to the attributes 
with the higher information gain.

1. Calculate entropy for dataset
2. For each attribute:
   Calculate entropy for all categorical values
   Calculate information gain for the current attribute
3. Find the feture with maximum information gain
4. Repeat

The `outlook` has the highest info gain of 0.24, so we will select it as 
the root node for the start level of splitting.
"""

import numpy as np
import pandas as pd
import pathlib
from sklearn import tree

# Dataset
DIR = pathlib.Path(__file__).resolve().parent
df = pd.read_csv(DIR / 'data/play_tennis.csv')

# Train data
X = df.drop(['play'], axis=1)
y = df['play']


# Total entropy (for current dataframe)
def dataset_entropy(df):
    E = 0
    N = df['play'].value_counts() # yes: 9, no: 5
    values = df['play'].unique()
    for v in values: # yes/no
        P = N[v]/len(df['play'])  # probability
        E += -P*np.log2(P)
    return E

# Entropy for each attribute
def attribute_entropy(df, attr):
    E = 0
    eps = np.finfo(float).eps   # machine epsilon for the float 
    targets = df.play.unique()
    values = df[attr].unique()
    for v in values: # cool/hot
        ent = 0
        for t in targets: # yes,no
            num = len(df[attr][df[attr] == v][df.play == t]) # numerator
            den = len(df[attr][df[attr] == v])
            fraction = num/(den + eps)
            ent += -fraction*np.log2(fraction + eps) # entropy for one feature
        E += -(den/len(df))*ent # sum of all entropies
    return abs(E)

# ---------------------------------------------------

# Find attribute with maximum information gain
def find_winner(df):
    IG = {}
    attributes = df.keys()[:-1]

    # Loop for attributes in dataframe and compute info gains
    for attr in attributes: 
        IG[attr] = dataset_entropy(df) - attribute_entropy(df, attr)
    winner = attributes[np.argmax(IG)] # maxim info gains
    return winner
# ---------------------------------------------------

# Construct the decision tree (dictionary)
def buildTree(df):
    tree = {}

    # Target column
    Class = df.keys()[-1] # play
    
    # Maximum info gain
    node = find_winner(df) # outlook
    tree[node] = {}

    # Distinct values
    values = np.unique(df[node]) # overcast/rain

    # Loop throw the attribute values
    for value in values:
        subtable = df[df[node] == value].reset_index(drop=True)
        attr_values, counts = np.unique(subtable[Class], return_counts=True)

        if len(counts) == 1: # pure subset
            tree[node][value] = attr_values[0]
        else:
            subtable = subtable.drop(node, axis=1)
            tree[node][value] = buildTree(subtable) # Recursive case
            
    return tree

decision_tree = buildTree(df)


# Print dictionary tree (recursion  in case of subtrees)
def print_tree(tree, attr=None, i=0):
    if not attr:
        attr = next(iter(tree)) # attrribute in the current tree node

    for key, subval in tree[attr].items():

        if isinstance(subval, str): # Base case
            print(i*" ", attr, "=", key, ":", subval)
            continue
        
        print(i*" ", attr, "=", key, ":")
        print_tree(subval, i=i+1) # Recursive

    return

# Predict unknow (only for cases included in the train dataset)
def predict(X, tree):
    key = next(iter(tree))
    val = X[key]
    subval = tree[key][val]

    if isinstance(subval, str): # Base case
        return subval
        
    subval = predict(X, subval) # Recursive
    return subval


print(decision_tree, "\n")
print_tree(decision_tree)

# Example usage
x = {'outlook': 'sunny', 'temp': 'mild', 'humidity': 'high', 'windy': False}
y = predict(x, decision_tree)
print("\nAttributes:", x)
print("Prediction:", y)

# Example usage 2
x = {'outlook': 'rainy', 'temp': 'mild', 'humidity': 'normal', 'windy': True}
y = predict(x, decision_tree)
print("\nAttributes:", x)
print("Prediction:", y)


"""
    {'outlook': {'overcast': 'yes', 'rainy': {'temp': {'cool': {'humidity': ...

    outlook = overcast: yes
    outlook = rainy:
      temp = cool:
        humidity = normal:
          windy = False: yes
          windy = True: no
      temp = mild:
        humidity = high:
          windy = False: yes
          windy = True: no
        humidity = normal: yes
    outlook = sunny:
      temp = cool: yes
      temp = hot: no
      temp = mild:
        humidity = high: no
        humidity = normal: yes

    Attributes: {
        'outlook': 'sunny', 
        'temp': 'mild', 
        'humidity': 'high', 
        'windy': False}
    Prediction: no

    Attributes: {
        'outlook': 'rainy', 
        'temp': 'mild', 
        'humidity': 'normal', 
        'windy': True}
    Prediction: yes
"""