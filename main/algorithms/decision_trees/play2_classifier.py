""" Decision Tree / Classifier (Play Tennis)

Given certain values for each of the attributes, the learned decision tree 
is able to give a clear answer if weather is suitable or not for tennis.
"""

import numpy as np
import pandas as pd
import pathlib
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
import graphviz

# Dataset
DIR = pathlib.Path(__file__).resolve().parent
df = pd.read_csv(DIR / 'data/play_tennis.csv')

# Encode data
for col in df.columns:
    df[col] = LabelEncoder().fit_transform(df[col])

# Train data
X = df.drop(['play'], axis=1) # remove column labeled `play`
y = df['play']

# Fitting the model
decision_tree = DecisionTreeClassifier(criterion='entropy')
decision_tree.fit(X, y)

# Output
dot_data = tree.export_graphviz(decision_tree, out_file=None, filled=True)
dot_graph = graphviz.Source(dot_data)
dot_graph.view()

tree_text = tree.export_text(decision_tree, feature_names=list(X.columns))
print(tree_text)

"""
|--- outlook <= 0.50
|   |--- class: 1
|--- outlook >  0.50
|   |--- humidity <= 0.50
|   |   |--- outlook <= 1.50
|   |   |   |--- windy <= 0.50
|   |   |   |   |--- class: 1
|   |   |   |--- windy >  0.50
|   |   |   |   |--- class: 0
|   |   |--- outlook >  1.50
|   |   |   |--- class: 0
|   |--- humidity >  0.50
|   |   |--- windy <= 0.50
|   |   |   |--- class: 1
|   |   |--- windy >  0.50
|   |   |   |--- outlook <= 1.50
|   |   |   |   |--- class: 0
|   |   |   |--- outlook >  1.50
|   |   |   |   |--- class: 1
"""