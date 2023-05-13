""" Knn / Movie recommendation system (user input)
"""

import pathlib
import pickle
from scipy import spatial
import operator

DIR = pathlib.Path(__file__).resolve().parent
with open(DIR / 'data/movies_processed.pkl', 'rb') as f:
    movies = pickle.load(f)

# Spatial distance between vectors
def similarity(movieId1, movieId2):
    a = movies.iloc[movieId1]
    b = movies.iloc[movieId2]
    d1 = spatial.distance.cosine(a['genres_bin'], b['genres_bin'])
    d2 = spatial.distance.cosine(a['cast_bin'], b['cast_bin'])
    d3 = spatial.distance.cosine(a['keywords_bin'], b['keywords_bin'])
    return d1 + d2 + d3

# Find the 10 most similiar movies
def predict_score():
    name = input('Enter a movie title: ')
    new_movie = movies[movies['original_title'].str.contains(name)]
    new_movie = new_movie.iloc[0].to_frame().T
    print('\nSelected Movie: ', new_movie.original_title.values[0], "\n")

    def getNeighbors(baseMovie, k_neighbors):
        distances = []

        for i, movie in movies.iterrows():
            if movie['new_id'] != baseMovie['new_id'].values[0]:
                d = similarity(baseMovie['new_id'].values[0], movie['new_id'])
                distances.append((movie['new_id'], d))
        distances.sort(key=operator.itemgetter(1))

        neighbors = []
        for i in range(k_neighbors):
            neighbors.append(distances[i])
        return neighbors

    neighbors = getNeighbors(new_movie, k_neighbors=10)
    for neighbor in neighbors:
        original_title = movies.iloc[neighbor[0]][0]
        genres = str(movies.iloc[neighbor[0]][1]).strip("[]").replace("'", "")
        rating = str(movies.iloc[neighbor[0]][2])
        print(original_title + " | Genres: " + genres + " | Rating: " + rating) 

predict_score()