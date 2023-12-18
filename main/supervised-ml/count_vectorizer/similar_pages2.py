""" Similar Pages - Count Vectorizer App

By using a Python script to handle the data processing and machine learning tasks, 
you can leverage Python's powerful libraries and then update your database.
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup
from icecream import ic
import numpy as np
import pymysql

# Database connection
conn = pymysql.connect(host='localhost', user='admin', password='password', db='minte9_refresh_v2')

# Fetch data
with conn.cursor() as c:
    query = """
        SELECT t1.page_id, COALESCE(t1.content_highlighted, t1.content) as content 
        FROM pages t1
        WHERE page_private=0
    """
    c.execute(query)
    data = c.fetchall()


# Data processing
vectorizer = CountVectorizer()

# Strip HTML tags from the text and store it in 'page_texts'
page_texts = [BeautifulSoup(text, 'html.parser').get_text() for _, text in data]
X = vectorizer.fit_transform(page_texts)

# print(page_texts[600])

# Calculate similarity
similarity_matrix = cosine_similarity(X)

# Print the similarity matrix and its shape
ic(similarity_matrix)
ic(np.array(similarity_matrix).shape)


# Get similar pages
with conn.cursor() as cursor:
    for i, page in enumerate(data):
        page_id = page[0]

        """
            Get top 3 similar pages, excluding the page itself
            Higher values do indicate higher similarity
            Numpy argsort returns the indices that would sort the array in ascending order
            We reverse indices array (higher values first)
        """
        similar_indices = np.argsort(similarity_matrix[i])[-4:-1][::-1]
        similar_pages = [data[idx][0] for idx in similar_indices]

        # Output results
        ic(i, page_id, similar_pages)

        if i >= 2: break

"""
    ic| similarity_matrix: array([[1.        , 0.09132484, 0.15395037, ..., 0.13038836, 0.08898476,
                                0.08533536],
                                [0.09132484, 1.        , 0.24806947, ..., 0.05701899, 0.04821475,
                                0.00850171],
                                [0.15395037, 0.24806947, 1.        , ..., 0.04884332, 0.06996956,
                                0.05483441],
                                ...,
                                [0.13038836, 0.05701899, 0.04884332, ..., 1.        , 0.17020715,
                                0.09925444],
                                [0.08898476, 0.04821475, 0.06996956, ..., 0.17020715, 1.        ,
                                0.11750014],
                                [0.08533536, 0.00850171, 0.05483441, ..., 0.09925444, 0.11750014,
                                1.        ]])
    ic| np.array(similarity_matrix).shape: (679, 679)
    ic| i: 0, page_id: 13, similar_pages: [1207, 32, 1159, 1369, 970]
    ic| i: 1, page_id: 14, similar_pages: [17, 1390, 66, 1207, 1388]
    ic| i: 2, page_id: 17, similar_pages: [1392, 151, 1207, 1388, 20]
"""