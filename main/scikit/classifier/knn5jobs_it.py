"""KNN IT Jobs classification (skills, languages)
KNeighborsClassifier class expects the input data to be a numeric array or matrix
Use the CountVectorizer class from scikit-learn, which converts a collection of text documents 
(such as a list of skills and programming languages) into a matrix of token counts
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

from knn5jobs_it_data import jobs
df = pd.DataFrame(jobs)
print(df)

def join_features(job):
    return ' '.join(job['skills'] + job['languages'])

jobs_string = [join_features(job) for job in jobs]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(jobs_string)
y = [job['title'] for job in jobs]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
knn = KNeighborsClassifier(n_neighbors=2) 
knn.fit(X_train, y_train)

predictions = knn.predict(X_test)
accuracy = knn.score(X_test, y_test)
print(f'Model accuracy: {accuracy:.2f}')

unknown_jobs = [
    {
        'skills': ['programming', 'web development', 'HTML', 'CSS', 'JavaScript'],
        'languages': ['Python', 'SQL', 'HTML', 'CSS', 'JavaScript']
    }, 
        # prediction: Full Stack Developer

    {
        'skills': ['programming', 'sql', 'javascript'],
        'languages': ['Php', 'Mysql', 'Python', 'SQL']
    },
        # prediction: Software Developer

    {
        'skills': ['programming', 'scripting', 'server maintenance'],
        'languages': ['Php', 'Java', 'Python', 'Javascript', 'Git']
    }
        # prediction: Full Stack Developer | Software Developer
]

for unknown in unknown_jobs:
    X = vectorizer.transform([join_features(unknown)])
    prediction = knn.predict(X)
    print(prediction) 