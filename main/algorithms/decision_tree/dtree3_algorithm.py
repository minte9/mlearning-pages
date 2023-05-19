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

# ------------------------------------------------------------------------------

# Entropy (total) for current dataframe
def dataset_entropy(df):
    entropy = 0
    targets = df.play

    for v in targets.unique(): # yes/no
        fraction = targets.value_counts()[v]/len(targets)
        entropy += -fraction*np.log2(fraction)
    return entropy

# Entropy (total) for one specific attribute
def attribute_entropy(df, attr):
    entropy = 0
    eps = np.finfo(float).eps # pi

    attr_targets = df.play.unique() # yes/no
    attr_values = df[attr].unique() # cool/hot

    for v in attr_values:
        attr_ent = 0

        for t in attr_targets:
            num = len(df[attr][df[attr] == v][df.play == t]) # numerator
            den = len(df[attr][df[attr] == v]) # denominator

            fraction = num/(den + eps)
            attr_ent += -fraction*np.log2(fraction + eps) # entropy for one feature

        entropy += -(den/len(df))*attr_ent # sum of all entropies
    return abs(entropy)

# Attribute with maxim info gains
def find_winner(df):
    attributes = df.keys()[:-1]
    total_entropy = dataset_entropy(df)

    # Loop for attributes in dataframe and compute info gains
    infogains = {}
    for attr in attributes: 
        infogains[attr] = total_entropy - attribute_entropy(df, attr)
    
    winner_attr = attributes[np.argmax(infogains)] # maxim info gains
    return winner_attr

# ------------------------------------------------------------------------------

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

# ------------------------------------------------------------------------------

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

# ------------------------------------------------------------------------------

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