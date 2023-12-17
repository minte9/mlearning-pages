""" Tokenizing (text)

Natural Language Toolkit (NLTK) has many powerfull text operations.
Tokenizing is the first step of turning the text into data.

Stop words (common words) contain little information value.
Nltk stopwords assumes the tokenized words are all lowercase.

Steamming reduces a word to its stem by removing affixes (gerunds).
For example, both 'traditional' and 'tradition' have 'tradit' as their stem.
"""

from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

s = "Today science is the technology of tomorrow. Tomorrow is today."

def tokenize_senteces(s):
    sentences = sent_tokenize(s)
    return sentences

def tokenize_words(s):
    words = word_tokenize(s)
    return words

def remove_stopwords(words):
    words = [w for w in words if w not in stopwords.words('english')]
    return words

def steming_words(words):
    porter = PorterStemmer()
    words = [porter.stem(w) for w in words]
    return words

# -------------------------------------------------------

# Output
print("Input text:", s); 

S = tokenize_senteces(s)
print("Sentences:", S)

words = tokenize_words(s)
print("Words:",  words)

words = remove_stopwords(words)
print("NoStopWords:", words)

words = steming_words(words)
print("Stemming", words)

"""
    Input text:  Today science is the technology of tomorrow. Tomorrow is today.
    Sentences:   ['Today science is the technology of tomorrow.', 'Tomorrow is today.']
    Words:       ['Today', 'science', 'is', 'the', 'technology', 'of', 'tomorrow', ...]
    NoStopWords: ['Today', 'science', 'technology', 'tomorrow', ...]
    Stemming     ['today', 'scienc', 'technolog', 'tomorrow', ...]
"""