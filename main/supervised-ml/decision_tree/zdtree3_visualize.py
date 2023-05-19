""" Decision Trees / Visualize

Visualize the tree using export_graphviz function.
This writes a .dot file which contain the text for storing graphs.
  pip install graphviz
  sudo apt-get install graphviz 

We can see that from the 142 samples that went to the right side, 
nearly all of them (132) end up in the leaf to the very right. 
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
import graphviz

import pathlib
DIR = pathlib.Path(__file__).resolve().parent

# Dataset
dataset = load_breast_cancer()

# Training and test data
X1, X2, y1, y2 = train_test_split(
    dataset.data, dataset.target, stratify=dataset.target, random_state=42)

# Pre-prunning
tree = DecisionTreeClassifier(max_depth=4, random_state=0)
tree.fit(X1, y1)

# Export tree
export_graphviz(tree, out_file="tree.dot", 
    class_names=["malignant", "benign"], feature_names=dataset.feature_names,
    impurity=False, filled=True)

# Visualize tree
dot_graph = graphviz.Source.from_file('tree.dot')
dot_graph.view()