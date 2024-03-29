""" Knn / Movie recommendation system (save processed)

We could save the cleaned and processed movies dataframe to a file, 
so that we don't have to repeat the data preparation steps every time. 
This could save a lot of time if you plan on running the code multiple times.
"""

import pathlib
import pandas as pd
import numpy as np
import json
from scipy import spatial
import operator
import pickle

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
movies['genres'] = movies['genres'].str.strip('[]').str.replace(' ', '').str.replace("'", '')
movies['cast'] = movies['cast'].str.strip('[]').str.replace(' ', '').str.replace("'", '')
movies['keywords'] = movies['keywords'].str.strip('[]').str.replace(' ', '').str.replace("'", '')

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
for val, index in zip(movies['cast'], movies.index): # select first 4 actors
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

# New clean dataset
new_id = list(range(0, movies.shape[0]))
movies['new_id'] = new_id
movies = movies[[
    'original_title', 'genres', 'vote_average',
    'genres_bin', 'cast_bin', 'keywords_bin', 'new_id',
]]

# ------------------------------------------------------------------------------------

with open(DIR / '_data/movies_processed.pkl', 'wb') as f:
    pickle.dump(movies, f)

# ------------------------------------------------------------------------------------