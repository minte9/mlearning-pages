""" TMDB / KNN classifier - Movie recommendation system

Using head() we can see that dataframe values are in JSON format.
We'll convert these columns into a format that can be easily read and interpreted. 
We decode json into a list.
"""

import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

# ------------------------------------------------------------------------------------

DIR = pathlib.Path(__file__).resolve().parent
movies = pd.read_csv(DIR / '_data/tmdb_5000_movies.csv') # dataframe
credits = pd.read_csv(DIR / '_data/tmdb_5000_credits.csv')

print("Movies head(): \n", movies.head())
print("Credits head(): \n", credits.head())
print("Movies columns: \n", movies.columns)
print("Credits columns: \n", credits.columns)

# Change columns values from json to string
def convert_json(df, feature):
    df[feature] = df[feature].apply(json.loads)
    for index, val in zip(df.index, df[feature]):
        lst = []
        for i in range(len(val)):
            lst.append(val[i]['name'])
        df.loc[index, feature] = str(lst)

convert_json(movies, 'genres')
convert_json(movies, 'keywords')
convert_json(movies, 'production_companies')
convert_json(credits, 'cast')
convert_json(credits, 'crew')

print("Movie 25: \n", movies.loc[25, ['genres', 'keywords', 'homepage']])
print("Credits 25: \n", credits.loc[25, ['cast', 'crew']])

# ------------------------------------------------------------------------------------

"""
    Movies head(): 
        budget                                             genres  ... vote_average  vote_count
    0  237000000  [{"id": 28, "name": "Action"}, {"id": 12, "nam...  ...          7.2       11800
    1  300000000  [{"id": 12, "name": "Adventure"}, {"id": 14, "...  ...          6.9        4500
    2  245000000  [{"id": 28, "name": "Action"}, {"id": 12, "nam...  ...          6.3        4466
    3  250000000  [{"id": 28, "name": "Action"}, {"id": 80, "nam...  ...          7.6        9106
    4  260000000  [{"id": 28, "name": "Action"}, {"id": 12, "nam...  ...          6.1        2124

    Credits head(): 
        movie_id  ...                                               crew
    0     19995  ...  [{"credit_id": "52fe48009251416c750aca23", "de...
    1       285  ...  [{"credit_id": "52fe4232c3a36847f800b579", "de...
    2    206647  ...  [{"credit_id": "54805967c3a36829b5002c41", "de...
    3     49026  ...  [{"credit_id": "52fe4781c3a36847f81398c3", "de...
    4     49529  ...  [{"credit_id": "52fe479ac3a36847f813eaa3", "de...

    Movies columns: 
    Index(['budget', 'genres', 'homepage', 'id', 'keywords', 'original_language',
        'original_title', 'overview', 'popularity', 'production_companies',
        'production_countries', 'release_date', 'revenue', 'runtime',
        'spoken_languages', 'status', 'tagline', 'title', 'vote_average',
        'vote_count'])

    Credits columns: 
    Index(['movie_id', 'title', 'cast', 'crew'])

    Movie 25: 
    genres                       ['Drama', 'Romance', 'Thriller']
    keywords    ['shipwreck', 'iceberg', 'ship', 'panic', 'tit...
    homepage                          http://www.titanicmovie.com
    
    Credits 25: 
    cast    ['Kate Winslet', 'Leonardo DiCaprio', 'Frances...
    crew    ['Mali Finn', 'James Horner', 'James Cameron',...
"""