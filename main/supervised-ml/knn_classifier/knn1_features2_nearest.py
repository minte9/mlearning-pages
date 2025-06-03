""" KNN classifier / Movie recomendation 2
Find the nearest neighbors to the movie the user watched.
"""

from sklearn.neighbors import NearestNeighbors

# Training dataset
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

# Movies watched by user
x_user = [[0.6, 7.9]]

# Train the KNN model
knn = NearestNeighbors(n_neighbors=3)
knn.fit(X)

# Find the nearest neighbors to the movie the user watched
_, indices = knn.kneighbors(x_user)

# Get the recommended movies
recommended_movies = [X[i] for i in indices.flatten()]

print("Movies watched by user:", x_user)
print("Recommended movies:", recommended_movies)

"""
    Movies watched by user: [[0.6, 7.9]]
    Recommended movies: [[0.6, 8.1], [0.2, 8.0], [0.2, 7.8]]
"""
