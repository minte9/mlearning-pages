## Handling Text

### Cleaning
 p94
Unstructured text data, like a book text or a tweet, is both the most interesting   
source of features and one of the most complex to handle.  

Most basic cleaning can be done with core string operations like strip, replace, split.  
In the real world we will most likely define a custom function.  

Translate is a Python popular method due to its blazing speed.  
To remove all punctuation we translate the P chars to None, effectively  
removing them, which is far faster than alternatives.  

### Tokenizing
 p98
When you want to break the text into individual words.  
Steamming reduces a word to its stem by removing affixes (gerunds).

### Tagging parts of speec
 p101
Use NLTK's pre-trained parts-of-speech tager.
Convert sentences into features for individual parts of speach.

### Bags of words
 p104
One of the most common model of transforming text into features.
Bag-of-words output a feature for every unique word in text data.

### Word importance
 p105
We compare the frequency of a word in a document with the frequency 
of the word in all other documents.

The more a word appears in a document, the more important is.
In contrast, if a word appears in many documents, it is less important 
to any individual document.

#

### References

[Installing NLTK Data](https://www.nltk.org/data.html)
[Sparce matrix](https://www.minte9.com/mlearning/numpy-matrices-1434)