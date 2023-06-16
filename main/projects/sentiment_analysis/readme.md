# Sentiment Analysis on Social Media

Collect a `small number of reviews` from amazon books.  
Manually `label each post` as positive or negative sentiment.   
Extract `text features` from the posts, such as word frequencies or TF-IDF values.  
Build a `dataset` with these features and labels.  
Train a `KNN classifier` to classify new posts based on their sentiment.  

### Data Preprocessing

Remove punctuation and numbers, convert text to lowercase, remove stop words.  

## Extracting text features

Extracting text features from posts, such as `word frequencies` or `TF-IDF values`,  
is a common task in natural language processing (NLP).

Word Frequencies: For each document (in this case, a social media post), you `count the occurrence of each word`. 
This results in a numerical representation of the text, where `each word becomes a feature`, 
and its corresponding `count becomes the feature value`.

TF-IDF (Term Frequency-Inverse Document Frequency): TF-IDF considers `not only the frequency of words in a document` 
but also their `importance in the entire dataset`. It assigns a weight to each word based on its occurrence in the document and inverse occurrence in the entire dataset. TF-IDF helps capture the relative importance of words across the documents.

Vectorization: After extracting the features, you need to convert them into `numerical vectors` that can be used as input for a machine learning algorithm. This process is known as vectorization. In the case of word frequencies or TF-IDF, you'll have a `sparse matrix` representation where `each row corresponds to a document` and each column corresponds to a unique word.

### KNN Classifier

The classifier will learn patterns in the data and be able to `classify new social media posts` based on their text features. There are various libraries in Python, such as scikit-learn, that provide built-in functions and utilities for text preprocessing, tokenization, feature extraction, and vectorization. These libraries can greatly simplify the implementation process.

Your system should be able to take an input text (such as a social media post, a review, or any other piece of text) and predict the sentiment associated with it. The sentiment could be classified as `positive, negative, or neutral`.

The size of the dataset can have an impact on the `performance and generalization` ability of your text classification model. 


### Sentiment140 Dataset

It contains 1,600,000 tweets extracted using the twitter api.
The tweets have been annotated  (0 = negative, 4 = positive) and they can be used to detect sentiment.


# Project Result

The programm is able to train a model using LogisticRegression, based on 100_000 samples from Sentiment140.  
With this model we can predict if a review on amazon books is positive or negative.  
Initial accuracy score is not so good, something around 60%  
Preprocessing like removing punctuation, numbers, whitespaces did help, increasing the accuracy around 70%  


### References

[Machine Learning with Python Cookbook](https://www.amazon.com/gp/product/B07BC3LFKT)
[Sentiment140 Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140)