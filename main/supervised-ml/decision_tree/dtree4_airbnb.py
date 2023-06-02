""" Decision Tree / Classifier (airbnb)

The DecisionTreeClassifier handles missing values in the dataset by either 
ignoring the samples with missing values during training or by imputing 
the missing values using strategies like mean, median, or most frequent value. 
By default, scikit-learn ignores the samples with missing values during training.

Separate the known instances and the unknown instance (?) from the dataset.
Train the decision tree classifier using the known instances.
The unknown predicted target value will be returned based on the learned patterns.
"""

import pathlib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# Dataset
DIR = pathlib.Path(__file__).resolve().parent
df = pd.read_csv(DIR / 'data/imobiliare.csv')

df_encoded = pd.DataFrame()
for col in df.columns:
    df_encoded[col] = LabelEncoder().fit_transform(df[col])

# Train data
X = df_encoded.drop(columns=["Nume", "Recomandat", "Link"])
Y = df_encoded['Recomandat']

# Last row
# x_new = X.iloc[-1:] # extract last row
# X = X[:-1] # remove last row (from features)
# Y = Y[:-1] # remove last row (from targets)

# Select row by value
indices = df[df['Recomandat'] == '?'].index
index = indices[0] if len(indices) else len(X)-1 # row with ? or last
x_new = X.loc[[index]]
X = X.drop(index)
Y = Y.drop(index)

# Fitting the model
dtree_model = DecisionTreeClassifier(random_state=42)
dtree_model.fit(X, Y)

output_tree = tree.export_text(dtree_model, feature_names=list(X.columns))

# Prediction
y_pred = dtree_model.predict(x_new)

targets = ['saptamana', 'weekend']
targets = ['da', 'nu']
print(df, "\n")
# print(df_encoded, "\n")
print(output_tree)
print("Unknown:"); print(df.loc[[index]]); # print(x_new, "\n")
print("Target classes: ", dtree_model.classes_)
print("Prediction:", y_pred, targets[y_pred[0]])


"""
		    nume   access-auto loc-parcare         parcare  ...        living   zona-lucru internet recomandat
	0     teilor          usor      gratis     '?', ra  ...  living-comun  zona-comuna     wifi    weekend
	3     3brazi          usor      gratis          usoara  ...   fara-living       nu-are     wifi    weekend
	4       tobo          usor      gratis          usoara  ...    cu-canapea        birou     wifi  saptamana
	5  x_unknown          usor      gratis          usoara  ...    cu-canapea        birou     wifi          ?

	   nume  access-auto  loc-parcare  parcare  imobil  ...  bucatarie.1  living  zona-lucru  internet  recomandat
	0     2            1            0        1       0  ...            2       1           1         0           1
	1     4            0            0        0       1  ...            0       0           1         0           2
	2     1            1            0        0       0  ...            1       3           2         0           2
	3     0            1            0        1       1  ...            3       2           1         0           2
	4     3            1            0        1       1  ...            3       0           0         0           1
	5     5            1            0        1       0  ...            2       0           0         0           0

	|--- camere <= 0.50
	|   |--- class: 1
	|--- camere >  0.50
	|   |--- class: 2

	Unknown:
	   access-auto  loc-parcare  parcare  imobil  ...  bucatarie.1  living  zona-lucru  internet
	5            1            0        1       0  ...            2       0           0         0

	Prediction: [1]
"""