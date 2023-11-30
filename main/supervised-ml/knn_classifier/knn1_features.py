from sklearn.neighbors import KNeighborsClassifier
from icecream import ic

# Training dataset
X = [[0,0], 
     [1,1], 
     [2,2], 
     [3,3]]    
y = [0, 1, 0, 1]

# Train the model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

# Make predictions for unknown
x_unknown = [1,2]        
y_pred = knn.predict([x_unknown])  

ic(x_unknown)
ic(y_pred);

"""
    ic| x_unknown: [1, 2]
    ic| y_pred: array([0])
"""