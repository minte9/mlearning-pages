""" Count Vectorizer
Represent texts as vectors and compute similarity.
The word `london` occurs 2 times in A and 1 time in B.
We can find the cos similarity between these two vectors.
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

text = ['London Paris London', 'Paris Paris London']
cv = CountVectorizer()
matrix = cv.fit_transform(text)

similarity_scores = cosine_similarity(matrix)

print("Features names: \n", cv.get_feature_names_out())
print("Matrix vectorized: \n", matrix)
print("Matrix array: \n", matrix.toarray(), "\n")
print ("Similarity scores: \n", similarity_scores)

"""
	Features names: 
	 ['london' 'paris']
	Matrix vectorized: 
	   (0, 0)       2
	  (0, 1)        1
	  (1, 0)        1
	  (1, 1)        2
	Matrix array: 
	 [[2 1]
	 [1 2]] 

	Similarity scores: 
	 [[1.  0.8]
	 [0.8 1. ]]
"""