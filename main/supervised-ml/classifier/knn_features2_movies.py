""" KNN classifier / Movie recomandation

We hava a dataset of 10 movies (ids) and 2 features (genre, rating).
We also have a movie that an user already saw (x_unknown).
We can predict what's the most recommended next movie for user.

In practice, it's not common to have a large number of tied neighbors, 
so the knn.predict() function usually returns a list containing 
one or a few recommended movies.
"""

from sklearn.neighbors import KNeighborsClassifier

# Training dataset (genre, rating)
X = [[0.2, 7.8],
     [0.8, 8.5],
     [0.5, 6.9],
     [0.6, 8.1],
     [0.3, 7.2],
     [0.9, 8.3],
     [0.7, 7.5],
     [0.4, 7.1],
     [0.1, 6.5],
     [0.2, 8.0]]

y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] # movie id

# Movie watched by user
x_user = [[0.6, 8.2]]

# Train the KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

# Predict the IDs of the next movies to watch
next_movie_ids = knn.predict(x_user)

print("Movie watched by user:", x_user)
print("Next movies to watch ids:", next_movie_ids)

"""
     Movie watched by user: [[0.6, 8.2]]
     Next movies to watch ids: [2]
"""