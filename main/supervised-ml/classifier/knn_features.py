""" KNN classifier / Features and label

We provide training dataset points (features) and label (target).
Next, we train the model using KNN classifier with k=3 (nearest neighbors).
Finally, we are able now to predict the label for a new (unknown) data point.   
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
y_pred = knn.predict([x_unknown])  

print("New point: x_unknown =", x_unknown)
print("Predicted label: y_pred =", y_pred)

"""
    New point: x_unknown = [1, 2]
    Predicted label: y_pred = [0]
"""