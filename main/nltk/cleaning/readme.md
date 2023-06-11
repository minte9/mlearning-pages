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