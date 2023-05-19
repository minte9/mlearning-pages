""" Decision Tree / Classifier (Play Tennis)

Given certain values for each of the attributes, the learned decision tree 
is able to give a clear answer if weather is suitable or not for tennis.

We have 4 features (outlook, temperature, humidity, windy) and the one target (play). 
Information gain it is used to identify which attribute provides more information.
The attribute with the highest gini is given the higher priority in the tree.

For example, by calculating the gini for `humidity` and `wind`, we would find that
`humidity` plays a more important role, so it is consider as a better classifier.
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

# Encode dataset
df_encoded = pd.DataFrame()
for col in df.columns:
    df_encoded[col] = LabelEncoder().fit_transform(df[col]) # sunny=2

# Train data
X = df_encoded.drop(['play'], axis=1) # remove column labeled `play`
y = df_encoded['play']

# Fitting the model
decision_tree = DecisionTreeClassifier(criterion='entropy')
decision_tree.fit(X, y)

# Output
dot_data = tree.export_graphviz(decision_tree, out_file=None, filled=True)
dot_graph = graphviz.Source(dot_data)
dot_graph.view()

tree_text = tree.export_text(decision_tree, feature_names=list(X.columns))

print(df, "\n")
print(df_encoded, "\n")
print(tree_text)

"""
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

        outlook  temp  humidity  windy  play
    0         2     1         0      0     0
    1         2     1         0      1     0
    2         0     1         0      0     1
    3         1     2         0      0     1
    4         1     0         1      0     1
    5         1     0         1      1     0
    6         0     0         1      1     1
    7         2     2         0      0     0
    8         2     0         1      0     1
    9         1     2         1      0     1
    10        2     2         1      1     1
    11        0     2         0      1     1
    12        0     1         1      0     1
    13        1     2         0      1     0

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