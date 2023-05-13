""" TMDB / KNN classifier - Movie recommendation system

Using head() we can see that dataframe values are in JSON format.
We'll convert these columns into a format that can be easily read and interpreted. 
We decode json into a list.

We merge the movies and credits dataframes and select the relevant columns.
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

# ------------------------------------------------------------------------------------

# pd.set_option('display.max_colwidth', 10)

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

print('Movies (merged and features selection): \n', movies.head())
print('Movies index: \n' , movies.index.values)
print('Movies columns: \n' , movies.columns.values)
print("Movie 25: \n", movies.loc[25])

# ------------------------------------------------------------------------------------

# Plot top genres
plt.subplots()
plt.title('Top genres')
lst = []
for i in movies['genres']:
    lst.extend(i)
ax = pd.Series(lst).value_counts()[:10].sort_values(ascending=True) \
        .plot.barh(width=0.9, color=sns.color_palette('hls', 10))

# Plot actors with highest appearences
plt.subplots()
plt.title('Actors with highest appearences')
lst = []
for i in movies['cast']:
    lst.extend(i)
ax = pd.Series(lst).value_counts()[:15].sort_values(ascending=True) \
        .plot.barh(width=0.9, color=sns.color_palette('muted', 40))

# Plot top keywords
plt.subplots()
plt.title('Top keywords')
lst = []
for i in movies['keywords']:
    lst.extend(i)
ax = pd.Series(lst).value_counts()[:10].sort_values(ascending=True) \
        .plot.barh(width=0.9, color=sns.color_palette('hls', 10))

plt.show()

# ------------------------------------------------------------------------------------

"""
    Movies (merged and features selection): 
            id  ...                                           keywords
    0   19995  ...  ['culture clash', 'future', 'space war', 'spac...
    1     285  ...  ['ocean', 'drug abuse', 'exotic island', 'east...
    2  206647  ...  ['spy', 'based on novel', 'secret agent', 'seq...
    3   49026  ...  ['dc comics', 'crime fighter', 'terrorist', 's...
    4   49529  ...  ['based on novel', 'mars', 'medallion', 'space...

    Movies index: 
     [0    1    2 ... 4800 4801 4802]

    Movies columns: 
     ['id' 'original_title' 'genres' 'cast' 'vote_average' 'keywords']

    Movie 25: 
    id                                                              597
    original_title                                              Titanic
    genres                                   [Drama, Romance, Thriller]
    cast              [KateWinslet, LeonardoDiCaprio, FrancesFisher,...
    vote_average                                                    7.5
    keywords          [shipwreck, iceberg, ship, panic, titanic, oce...
    Name: 25, dtype: object
"""