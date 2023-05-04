""" KNN Supervised machine learning (ML) algorithm
K is the number of nearest neighbors to be used.

We provide training dataset points and labels and
createa a KNN classifier with K=3

Now, we can predict the label of a new data point.   
"""

from sklearn.neighbors import KNeighborsClassifier

X = [[0,0], [1,1], [2,2], [3,3]]    
y = [0, 1, 0, 1]

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)
            
prediction = knn.predict([[1,2]])   
print(prediction) # [0]