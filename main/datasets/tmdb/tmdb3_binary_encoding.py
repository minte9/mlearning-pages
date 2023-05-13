""" TMDB / KNN classifier - Movie recommendation system

Using head() we can see that dataframe values are in JSON format.
Convert these columns into a format that can be easily read and interpreted. 

Decode json into a list.
Merge the movies and credits dataframes and select the relevant columns.

Classify movies according to their genres (encoding for multiple labels).
Classify movies according to their actors (with highest contribution to the movie).

Luckily, the sequence of the actors in json is according to the actors' contribution.
We select the main 4 actors from each movie.
"""

import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

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
convert_json(movies, 'production_companies')
convert_json(credits, 'cast')
convert_json(credits, 'crew')

# Merge csv files and select the relevant columns
movies = movies.merge(credits, left_on='id', right_on='movie_id', how='left')
movies = movies[['id', 'original_title', 'genres', 'cast', 'vote_average', 'keywords']]

# Clean the columns
movies['genres'] = movies['genres'].str.strip('[]').str.replace(' ', '').str.replace("'", '')
movies['cast'] = movies['cast'].str.strip('[]').str.replace(' ', '').str.replace("'", '')
movies['keywords'] = movies['keywords'].str.strip('[]').str.replace(' ', '').str.replace("'", '')

# Get lists
movies['genres'] = movies['genres'].str.split(',')
movies['cast'] = movies['cast'].str.split(',')
movies['keywords'] = movies['keywords'].str.split(',')

# ------------------------------------------------------------------------------------

# Classify movies according to their genres (encoding for multiple labels)
genreList = []
for _, row in movies.iterrows():
    genres = row['genres']
    for v in genres:
        if v not in genreList:
            genreList.append(v)

# Binary values list (1 or 0 / genre is present or not in movie)
def binary(movie_genres):
    lst = []
    for v in genreList:
        if v in movie_genres:
            lst.append(1)
        else:
            lst.append(0)
    return lst

# New column that will hold the binary values    
movies['genres_bin'] = movies['genres'].apply(lambda x: binary(x))

# ------------------------------------------------------------------------------------

# Classify movies according to their actors (with highest contribution to the movie)
for val, index in zip(movies['cast'],movies.index):
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

# Binary values list (1 or 0 / actor is present or not in movie)
def binary(movie_actors):
    lst = []
    for v in castList:
        if v in movie_actors:
            lst.append(1)
        else:
            lst.append(0)
    return lst

# New column that will hold the binary values 
movies['cast_bin'] = movies['cast'].apply(lambda x: binary(x))
movies['cast_bin'].head()

# ------------------------------------------------------------------------------------

# Classify movies according to their keywords (encoding for multiple labels)
keywordsList = []
for _, row in movies.iterrows():
    keywords = row['keywords']
    for v in keywords:
        if v not in keywordsList:
            keywordsList.append(v)

# Binary values list (1 or 0 / keyword is present or not in movie)
def binary(movie_keywords):
    lst = []
    for v in keywordsList:
        if v in movie_keywords:
            lst.append(1)
        else:
            lst.append(0)
    return lst

# New column that will hold the binary values    
movies['keywords_bin'] = movies['keywords'].apply(lambda x: binary(x))

# ------------------------------------------------------------------------------------

print("Genre list [0:10]: \n", genreList[:5])
print("Movies genres binary: \n", movies['genres_bin'].head(), "\n")

print("Cast list [0:10]: \n", genreList[:5])
print("Movies cast binary: \n", movies['cast_bin'].head(), "\n")

print("Keywords list [0:10]: \n", keywordsList[:5])
print("Movies keywords binary: \n", movies['keywords_bin'].head(), "\n")

print("Movie 25 cast: \n", movies.loc[25])

"""
    Genre list [0:10]: 
    ['Action', 'Adventure', 'Fantasy', 'ScienceFiction', 'Crime']
    Movies genres binary: 
    0    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...
    1    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...
    2    [1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...
    3    [1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, ...
    4    [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...

    Cast list [0:10]: 
    ['Action', 'Adventure', 'Fantasy', 'ScienceFiction', 'Crime']
    Movies cast binary: 
    0    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...
    1    [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, ...
    2    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, ...
    3    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, ...
    4    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...

    Keywords list [0:10]: 
    ['cultureclash', 'future', 'spacewar', 'spacecolony', 'society']
    Movies keywords binary: 
    0    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
    1    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...
    2    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...
    3    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...
    4    [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, ...

    Movie 25 cast: 
    id                                                              597
    original_title                                              Titanic
    genres                                   [Drama, Romance, Thriller]
    cast              [KateWinslet, LeonardoDiCaprio, FrancesFisher,...
    vote_average                                                    7.5
    keywords          [shipwreck, iceberg, ship, panic, titanic, oce...
    genres_bin        [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, ...
    cast_bin          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...
    keywords_bin      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...
"""