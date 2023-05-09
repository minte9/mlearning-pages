""" KNN classifier / Features and label

We provide training dataset points (features) and label (target).
Next, we train the model using KNN classifier with k=3 (nearest neighbors to be used).
Finally, we are able now to predict the label of a new data point.   
"""

from sklearn.neighbors import KNeighborsClassifier

# Training dataset
X = [[0,0], 
     [1,1], 
     [2,2], 
     [3,3]]    
y = [0, 1, 0, 1]

# Train the model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

# Make predictions
x_unknown = [1,2]        
prediction = knn.predict([x_unknown])   
print("Label prediction for", x_unknown, "=", prediction)

"""
    Label prediction for [1, 2] = [0]
"""