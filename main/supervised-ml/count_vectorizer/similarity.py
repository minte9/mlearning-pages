
import numpy as np
import pandas as pd
import pickle
from scipy import spatial

import pathlib
DIR = pathlib.Path(__file__).resolve().parent

# Load the preprocessed data
with open(DIR / 'data/movies_processed.pkl', 'rb') as f:
    movies = pickle.load(f)

def similarity(a, b):
    d1 = spatial.distance.cosine(a['genres_bin'], b['genres_bin'])
    d2 = spatial.distance.cosine(a['cast_bin'], b['cast_bin'])
    d3 = spatial.distance.cosine(a['keywords_bin'], b['keywords_bin'])
    return d1 + d2 + d3

def predict_similar_movies(movie_title):
    name = "Avatar" # input('Enter a movie title: ')
    new_movie = movies[movies['original_title'].str.contains(movie_title, case=False, regex=False)].iloc[0]
    print('\nSelected Movie: ', new_movie.original_title, "\n")

    # Convert new_movie to a series for efficiency
    new_movie_series = new_movie[['genres_bin', 'cast_bin', 'keywords_bin']]

    # Calculate distances in a vectorized way
    distances = movies.apply(lambda x: similarity(new_movie_series, x) 
                             if x['new_id'] != new_movie['new_id'] else np.inf, axis=1)

    # Get the 10 most similar movies
    nearest_neighbors = distances.nsmallest(11).index

    for neighbor_idx in nearest_neighbors:
        if neighbor_idx != new_movie.name:  # Exclude the selected movie itself
            neighbor = movies.loc[neighbor_idx]
            print(f"{neighbor['original_title']} | Genres: {neighbor['genres']} | Rating: {neighbor['vote_average']}")

predict_similar_movies("Avatar")
predict_similar_movies("Titanic")
predict_similar_movies("Star Trek")

"""
    Selected Movie:  Avatar 

    Star Trek Into Darkness | Genres: ['Action', 'Adventure', 'ScienceFiction'] | Rating: 7.4
    Jupiter Ascending | Genres: ['ScienceFiction', 'Fantasy', 'Action', 'Adventure'] | Rating: 5.2
    Guardians of the Galaxy | Genres: ['Action', 'ScienceFiction', 'Adventure'] | Rating: 7.9
    Clash of the Titans | Genres: ['Adventure', 'Fantasy', 'Action'] | Rating: 5.6
    John Carter | Genres: ['Action', 'Adventure', 'ScienceFiction'] | Rating: 6.1
    Pirates of the Caribbean: On Stranger Tides | Genres: ['Adventure', 'Action', 'Fantasy'] | Rating: 6.4
    The Fifth Element | Genres: ['Adventure', 'Fantasy', 'Action', 'Thriller', 'ScienceFiction'] | Rating: 7.3
    The Time Machine | Genres: ['ScienceFiction', 'Adventure', 'Action'] | Rating: 5.8
    Superman Returns | Genres: ['Adventure', 'Fantasy', 'Action', 'ScienceFiction'] | Rating: 5.4
    Man of Steel | Genres: ['Action', 'Adventure', 'Fantasy', 'ScienceFiction'] | Rating: 6.5
    X-Men: Days of Future Past | Genres: ['Action', 'Adventure', 'Fantasy', 'ScienceFiction'] | Rating: 7.5

    Selected Movie:  Titanic 

    Revolutionary Road | Genres: ['Drama', 'Romance'] | Rating: 6.7
    The Great Gatsby | Genres: ['Drama', 'Romance'] | Rating: 7.3
    Romeo + Juliet | Genres: ['Drama', 'Romance'] | Rating: 6.7
    Iris | Genres: ['Drama', 'Romance'] | Rating: 6.2
    The Beach | Genres: ['Drama', 'Adventure', 'Romance', 'Thriller'] | Rating: 6.3
    All the King's Men | Genres: ['Drama', 'Thriller'] | Rating: 5.7
    The Reader | Genres: ['Drama', 'Romance'] | Rating: 7.2
    Sense and Sensibility | Genres: ['Drama', 'Romance'] | Rating: 7.2
    Little Children | Genres: ['Romance', 'Drama'] | Rating: 6.9
    What's Eating Gilbert Grape | Genres: ['Romance', 'Drama'] | Rating: 7.5
    Cruel Intentions | Genres: ['Drama', 'Romance', 'Thriller'] | Rating: 6.6

    Selected Movie:  Star Trek Into Darkness 

    Star Trek Beyond | Genres: ['Action', 'Adventure', 'ScienceFiction'] | Rating: 6.6
    Star Trek | Genres: ['ScienceFiction', 'Action', 'Adventure'] | Rating: 7.4
    Avatar | Genres: ['Action', 'Adventure', 'Fantasy', 'ScienceFiction'] | Rating: 7.2
    Transformers: Age of Extinction | Genres: ['ScienceFiction', 'Action', 'Adventure'] | Rating: 5.8
    Guardians of the Galaxy | Genres: ['Action', 'ScienceFiction', 'Adventure'] | Rating: 7.9
    Captain America: Civil War | Genres: ['Adventure', 'Action', 'ScienceFiction'] | Rating: 7.1
    Oblivion | Genres: ['Action', 'ScienceFiction', 'Adventure', 'Mystery'] | Rating: 6.4
    Pacific Rim | Genres: ['Action', 'ScienceFiction', 'Adventure'] | Rating: 6.7
    Avengers: Age of Ultron | Genres: ['Action', 'Adventure', 'ScienceFiction'] | Rating: 7.3
    Riddick | Genres: ['ScienceFiction', 'Action', 'Thriller'] | Rating: 6.2
    Ender's Game | Genres: ['ScienceFiction', 'Action', 'Adventure'] | Rating: 6.6
"""