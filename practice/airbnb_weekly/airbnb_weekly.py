""" Classifier models (airbnb)

Separate the known instances and the unknown instance (?) from the dataset.
Train the decision tree classifier using the known instances.
The unknown predicted target value will be returned based on the learned patterns.
"""

import pathlib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# Dataset
DIR = pathlib.Path(__file__).resolve().parent
df = pd.read_csv(DIR / 'data/airbnb.csv')

# -------------------------------------------------------------------------

# Encode labels
df_encoded = pd.DataFrame()
for col in df.columns:
    df_encoded[col] = LabelEncoder().fit_transform(df[col])

# Train data
X = df_encoded.drop(columns=["Nume", "Recomandat", "Link", "Pret"])
Y = df_encoded['Recomandat']
targets = ['saptamana', 'weekend']
unknown = len(X)-1 # last row

# Select unknown (?)
indices = df[df['Recomandat'] == '?'].index
if len(indices):
    targets = ['?', 'saptamana', 'weekend']
    unknown = indices[0]
    
x_new = X.iloc[[unknown]]

# Remove searched row from learning
X = X.drop(unknown)
Y = Y.drop(unknown)

print(df, "\n")
print(df_encoded, "\n")

# Train and test data
X1, X2, Y1, Y2 = train_test_split(X, Y, test_size=0.1, random_state=42)

# -------------------------------------------------------------------------

# Decistion Tree
model = DecisionTreeClassifier(random_state=42, max_depth=3)
model.fit(X1, Y1)
score1 = model.score(X1, Y1)
score2 = model.score(X2, Y2)
y_pred = model.predict(x_new)
print(tree.export_text(model, feature_names=list(X.columns)))
print("Unknown:"); print(df.loc[[unknown]])
print("Target classes: ", model.classes_, "\n")
print("Decision Tree:", y_pred, targets[y_pred[0]])
print("Score:", round(score1, 2), "/", round(score2, 2), "\n")

# -------------------------------------------------------------------------

# Random Forest
model = RandomForestClassifier(random_state=42)
model.fit(X1, Y1)
score1 = model.score(X1, Y1)
score2 = model.score(X2, Y2)
y_pred = model.predict(x_new)
print("Random Forest:", y_pred, targets[y_pred[0]])
print("Score:", round(score1, 2), "/", round(score2, 2), "\n")

# ----------------------------------------------------------------------

# Gradient Boosting
model = GradientBoostingClassifier(random_state=0, max_depth=3)
model.fit(X1, Y1)
score1 = model.score(X1, Y1)
score2 = model.score(X2, Y2)
y_pred = model.predict(x_new)
print("Gradient Boosting:", y_pred, targets[y_pred[0]])
print("Score:", round(score1, 2), "/", round(score2, 2), "\n")

# ----------------------------------------------------------------------

# Logistic Regression
model = LogisticRegression(random_state=42, class_weight='balanced')
model.fit(X1, Y1)
score1 = model.score(X1, Y1)
score2 = model.score(X2, Y2)
y_pred = model.predict(x_new)
print("Logistic Regression:", y_pred, targets[y_pred[0]])
print("Score:", round(score1, 2), "/", round(score2, 2), "\n")