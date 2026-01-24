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

# Import dataset from file
import sys
sys.dont_write_bytecode = True # no .pyc files
from data import jobs

# Training dataframe
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
print(f'Model accuracy: {accuracy:.2f} \n')

unknown_jobs = [
    {
        'skills':    ['programming', 'web development', 'HTML', 'CSS', 'JavaScript'],
        'languages': ['Python', 'SQL', 'HTML', 'CSS', 'JavaScript']
    }, 
    {
        'skills':    ['programming', 'sql', 'javascript'],
        'languages': ['Php', 'Mysql', 'Python', 'SQL']
    },
    {
        'skills':    ['programming', 'scripting', 'server maintenance'],
        'languages': ['Php', 'Java', 'Python', 'Javascript', 'Git']
    }
]

for item in unknown_jobs:
    X = vectorizer.transform([join_features(item)])
    prediction = knn.predict(X)
    print("Prediction for", item['languages'], "\n", prediction, "\n")

"""
		                    title  ...                                         languages
	0          Software Developer  ...                           [Java, Python, C++, C#]
	1          Software Developer  ...                                            [Java]
	2          Software Developer  ...                                          [Python]
	3          Software Developer  ...                                             [C++]
	4          Software Developer  ...                                              [C#]
	5       Systems Administrator  ...                  [Linux, Bash, Python, Ruby, SQL]
	6       Systems Administrator  ...                                            [Bash]
	7       Systems Administrator  ...                                          [Python]
	8       Systems Administrator  ...                                            [Ruby]
	9       Systems Administrator  ...                                             [SQL]
	10             Data Scientist  ...                                  [Python, R, SQL]
	11             Data Scientist  ...                                          [R, SQL]
	12             Data Scientist  ...                                          [Python]
	13            DevOps Engineer  ...                      [Python, Bash, Ruby, Docker]
	14            DevOps Engineer  ...                                            [Bash]
	15            DevOps Engineer  ...                                            [Ruby]
	16            DevOps Engineer  ...                                          [Docker]
	17       Full Stack Developer  ...  [HTML, CSS, JavaScript, Python, Java, PHP, Ruby]
	18       Full Stack Developer  ...                           [HTML, CSS, JavaScript]
	19       Full Stack Developer  ...                                       [Java, PHP]
	20       Full Stack Developer  ...                                    [Python, Ruby]
	21           Mobile Developer  ...                [Java, Kotlin, Swift, Objective-C]
	22           Mobile Developer  ...                                     [Objective-C]
	23           Mobile Developer  ...                                     [Java, Swift]
	24  Cloud Solutions Architect  ...        [Python, Java, Bash, Ruby, C++, Terraform]
	25          Security Engineer  ...                          [Python, Bash, SQL, C++]
	26            Software Tester  ...                   [Java, Python, JavaScript, SQL]
	27           Technical Writer  ...                             [HTML, CSS, Markdown]

	[28 rows x 3 columns]
	Model accuracy: 0.83

	Prediction for ['Python', 'SQL', 'HTML', 'CSS', 'JavaScript'] 
	 ['Full Stack Developer']

	Prediction for ['Php', 'Mysql', 'Python', 'SQL'] 
	 ['Software Developer']

	Prediction for ['Php', 'Java', 'Python', 'Javascript', 'Git'] 
	 ['Software Developer']
"""