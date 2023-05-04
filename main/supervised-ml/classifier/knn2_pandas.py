"""KNN classification / Fruit example

Dataset contains heights, widths and labels (fruit name).
The algorithm teach a function to map any combination 
in order to predict an unknown fruit name.
"""

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

data = {
  'height': [
    3.91, 7.09, 10.48, 9.21, 7.95, 7.62, 7.95, 4.69, 7.50, 7.11, 
    4.15, 7.29, 8.49, 7.44, 7.86, 3.93, 4.40, 5.5, 8.10, 8.69
  ], 
  'width': [
     5.76, 7.69, 7.32, 7.20, 5.90, 7.51, 5.32, 6.19, 5.99, 7.02, 
     5.60, 8.38, 6.52, 7.89, 7.60, 6.12, 5.90, 4.5, 6.15, 5.82
  ],
  'fruit': [
    'Mandarin', 'Apple', 'Lemon', 'Lemon', 'Lemon', 'Apple', 'Mandarin', 
    'Mandarin', 'Lemon', 'Apple', 'Mandarin', 'Apple', 'Lemon', 'Apple', 
    'Apple', 'Apple', 'Mandarin', 'Lemon', 'Lemon', 'Lemon'
  ]
} 
# Transform dataset into a DataFrame
df = pd.DataFrame(data) 

X = df[['height', 'width']].values
y = df.fruit.values

# Train the model
knn = KNeighborsClassifier(n_neighbors=3) 
knn.fit(X, y)

# Make predictions
new_item  = [9, 3]
new_items = [[9, 3], [4, 5], [2, 5], [8, 9], [5, 7]]

prediction  = knn.predict([new_item])
predictions = knn.predict(new_items)

print(df, "\n")
print("Prediction label for item", new_item, "\n", prediction, "\n")
print("Precition labels for items", new_items, "\n", predictions, "\n") 

"""
		height  width     fruit
	0     3.91   5.76  Mandarin
	1     7.09   7.69     Apple
	2    10.48   7.32     Lemon
	3     9.21   7.20     Lemon
	4     7.95   5.90     Lemon
	5     7.62   7.51     Apple
	6     7.95   5.32  Mandarin
	7     4.69   6.19  Mandarin
	8     7.50   5.99     Lemon
	9     7.11   7.02     Apple
	10    4.15   5.60  Mandarin
	11    7.29   8.38     Apple
	12    8.49   6.52     Lemon
	13    7.44   7.89     Apple
	14    7.86   7.60     Apple
	15    3.93   6.12     Apple
	16    4.40   5.90  Mandarin
	17    5.50   4.50     Lemon
	18    8.10   6.15     Lemon
	19    8.69   5.82     Lemon

	Prediction label for item [9, 3]
	 ['Lemon']

	Precition labels for items [[9, 3], [4, 5], [2, 5], [8, 9], [5, 7]]
	 ['Lemon' 'Mandarin' 'Mandarin' 'Apple' 'Mandarin']
"""