""" Knn / Movie recommendation system (scikit)

Combine relevant features and compute similarity score.
Sort by similarity score x[1] in descending order.
Exclude the first element, which is the movie itself.
"""

import pathlib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DIR = pathlib.Path(__file__).resolve().parent
movies = pd.read_csv(DIR / 'data/movies_dataset2.csv')

# New combined feature
def combine_features(row):
    return str(row['genres']) + " " + str(row['cast']) + " " + str(row['keywords'])

movies['combined_features'] = movies.apply(combine_features, axis=1)

# Similarity
cv = CountVectorizer()
count_matrix = cv.fit_transform(movies['combined_features'])
cosine_similarity = cosine_similarity(count_matrix)

# Find the 10 most similiar movies
def predict_movies(name):
    index = movies[movies['original_title'].str.contains(name)].index[0]
    similar_movies = list(enumerate(cosine_similarity[index]))

    # Sort by score in descending order
    similar_sorted = sorted(similar_movies, key=lambda x: x[1], reverse=True)[1:]

    # Output results
    for neighbor in similar_sorted[:10]:
        movie_index = neighbor[0]
        data = movies.iloc[movie_index]

        original_title = data['original_title']
        genres = data['genres']
        rating = str(data['vote_average'])
        print(original_title + " | " + genres + " | Rating: " + rating) 
        
predict_movies('Avatar')

"""
    Guardians of the Galaxy | Action Science Fiction Adventure | Rating: 7.9
    Star Trek Into Darkness | Action Adventure Science Fiction | Rating: 7.4
    Star Trek Beyond | Action Adventure Science Fiction | Rating: 6.6
    Alien | Horror Action Thriller Science Fiction | Rating: 7.9
    Star Wars: Clone Wars (Volume 1) | Action ... Fiction | Rating: 8.0
    Planet of the Apes | Thriller Science Fiction Action Adventure | Rating: 5.6
    Moonraker | Action Adventure Thriller Science Fiction | Rating: 5.9
    Galaxy Quest | Comedy Family Science Fiction | Rating: 6.9
    Gravity | Science Fiction Thriller Drama | Rating: 7.3
    Jupiter Ascending | Science Fiction Fantasy Action Adventure | Rating: 5.2
"""