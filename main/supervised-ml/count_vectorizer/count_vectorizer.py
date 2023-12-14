from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from icecream import ic

# Sample text strings
A = 'London Paris London'
B = 'Paris Paris London'

# Create an instance of CountVectorizer
cv = CountVectorizer()

# Convert the text strings into a frequency matrix
# Each unique word becomes a feature (column) in the matrix
matrix = cv.fit_transform([A, B])

# Calculate the cosine similarity between the vectors
# Since there are two documents, this results in a 2x2 matrix
# Diagonal elements are the self-similarity scores (1.0)
# Off-diagonal elements are the cross-document similarity scores
similarity_scores = cosine_similarity(matrix)

ic(cv.get_feature_names_out())
ic(matrix)
ic(matrix.toarray())
ic(similarity_scores)

"""
	ic| cv.get_feature_names_out(): array(['london', 'paris'], dtype=object)
	ic| matrix: <2x2 sparse matrix of type '<class 'numpy.int64'>'
					with 4 stored elements in Compressed Sparse Row format>
	ic| matrix.toarray(): array([[2, 1],
								[1, 2]])
	ic| similarity_scores: array([[1. , 0.8],
								[0.8, 1. ]])
"""