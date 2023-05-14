""" Knn / Movie recommendation system

Convert json columns into list to read and interpreted them easily. 
Merge the movies and credits dataframes and select the relevant columns.
Classify movies according to their genres (encoding for multiple labels).
New binary values which represents the presense of absence of a feature.
The vectors formed using binary values are called `one-hot` encoded vectors.
"""

import pathlib
import pandas as pd
import numpy as np
import json
from scipy import spatial
import operator

DIR = pathlib.Path(__file__).resolve().parent
movies = pd.read_csv(DIR / 'data/tmdb_5000_movies.csv')
credits = pd.read_csv(DIR / 'data/tmdb_5000_credits.csv')

# Change columns values from json to string
def convert_json(df, col):
    df[col] = df[col] \
                .apply(json.loads) \
                .apply(lambda x: [i['name'] for i in x]) \
                .apply(lambda x: str(x))

convert_json(movies, 'genres')
convert_json(movies, 'keywords')
convert_json(credits, 'cast')

# Merge csv files and select the relevant columns
movies = movies.merge(credits, left_on='id', right_on='movie_id', how='left')
movies = movies[['id', 'original_title', 'genres', 'cast', 'vote_average', 'keywords']]

# Clean the columns
movies['genres'] = \
    movies['genres'].str.strip('[]').str.replace(' ', '').str.replace("'", '')
movies['cast'] = \
    movies['cast'].str.strip('[]').str.replace(' ', '').str.replace("'", '')
movies['keywords'] = \
    movies['keywords'].str.strip('[]').str.replace(' ', '').str.replace("'", '')

# Get lists
movies['genres'] = movies['genres'].str.split(',')
movies['cast'] = movies['cast'].str.split(',')
movies['keywords'] = movies['keywords'].str.split(',')

# Classify movies by genres
genreList = []
for _, row in movies.iterrows():
    genres = row['genres']
    for v in genres:
        if v not in genreList:
            genreList.append(v)

def binary_genres(movie_genres):
    lst = []
    for v in genreList:
        if v in movie_genres:
            lst.append(1)
        else:
            lst.append(0)
    return lst

# Classify movies by actors 
for val, index in zip(movies['cast'],movies.index): # select first 4 actors
    lst = val[:4]
    movies.loc[index, 'cast'] = str(lst)
movies['cast'] = movies['cast'].str.strip('[]').str.replace(" '",'').str.replace("'",'')
movies['cast'] = movies['cast'].str.split(',')

castList = []
for index, row in movies.iterrows():
    cast = row["cast"]
    for i in cast:
        if i not in castList:
            castList.append(i)

def binary_cast(movie_actors):
    lst = []
    for v in castList:
        if v in movie_actors:
            lst.append(1)
        else:
            lst.append(0)
    return lst

# Classify movies by keywords
keywordsList = []
for _, row in movies.iterrows():
    keywords = row['keywords']
    for v in keywords:
        if v not in keywordsList:
            keywordsList.append(v)

def binary_keywords(movie_keywords):
    lst = []
    for v in keywordsList:
        if v in movie_keywords:
            lst.append(1)
        else:
            lst.append(0)
    return lst

# New binary columns
movies['genres_bin'] = movies['genres'].apply(lambda x: binary_genres(x))
movies['cast_bin'] = movies['cast'].apply(lambda x: binary_cast(x)) 
movies['keywords_bin'] = movies['keywords'].apply(lambda x: binary_keywords(x))

# Spatial distance between vectors
def similarity(movieId1, movieId2):
    a = movies.iloc[movieId1]
    b = movies.iloc[movieId2]

    d1 = spatial.distance.cosine(a['genres_bin'], b['genres_bin'])
    d2 = spatial.distance.cosine(a['cast_bin'], b['cast_bin'])
    d3 = spatial.distance.cosine(a['keywords_bin'], b['keywords_bin'])

    return d1 + d2 + d3

# New clean dataset
new_id = list(range(0, movies.shape[0]))
movies['new_id'] = new_id
movies = movies[[
    'original_title', 'genres', 'vote_average',
    'genres_bin', 'cast_bin', 'keywords_bin', 'new_id',
]]

# Find the 10 most similiar movies
def predict_movies(name):
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

predict_movies('Godfather')

# ------------------------------------------------------------------------------------

"""
    Selected Movie:  The Godfather: Part III 

    The Godfather: Part II | Genres: Drama, Crime | Rating: 8.3
    Donnie Brasco | Genres: Crime, Drama, Thriller | Rating: 7.4
    The Son of No One | Genres: Drama, Thriller, Crime | Rating: 4.8
    The Godfather | Genres: Drama, Crime | Rating: 8.4
    Absolute Power | Genres: Crime, Drama, Thriller | Rating: 6.4
    The Devil's Own | Genres: Crime, Thriller, Drama | Rating: 5.9
    We Own the Night | Genres: Drama, Crime, Thriller | Rating: 6.5
    The Counselor | Genres: Thriller, Crime, Drama | Rating: 5.0
    The Rainmaker | Genres: Drama, Crime, Thriller | Rating: 6.7
    Righteous Kill | Genres: Action, Crime, Drama, Thriller | Rating: 5.9
"""