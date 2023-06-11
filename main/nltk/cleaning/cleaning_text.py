""" Cleaning Text

Most text data will not be cleaned before we can use it to build features.
In the real world we will most likely define a custom cleaning function.

Translate is a Python method popular due to its blazing speed.
To remove all punctuation we translate the P chars to None, effectively 
removing them, which is far faster than alternatives.
"""

import string
import re
import sys
import unicodedata

A = [
    "  A b. C d  ", 
    "  A b. C d  "]

X = [
    'A!!! B. C....',
    '100% B! #C ?!']

# Strip whitespaces, remove periods, apply function
B = [s.strip() for s in A]
C = [s.replace(".", "") for s in B]
D = [s.lower() for s in C]

# Apply custom function
def transform(s: str) -> str:
    return s.upper()
E = [transform(s) for s in D]

# Apply regex
def regex_tranform(s: str) -> str:
    return re.sub(r"[a-zA-Z]", "x", s)
F = [regex_tranform(s) for s in E]

# Remvove punctuation
P = dict()
for i in range(sys.maxunicode):
    if unicodedata.category(chr(i)).startswith('P'):
        P[i] = None
G = [s.translate(P) for s in X]

# Remvove punctuation (second method)
P = dict.fromkeys(
      i for i in range(sys.maxunicode) 
        if unicodedata.category(chr(i)).startswith('P')
)
G = [s.translate(P) for s in X]

# Output
print("Strip whitespace:", B)
print("Periods removed:", C)
print("Apply lower:", D)
print("Apply custom:", E)
print("Apply regex:", F)
print("Remove punctuations:", G)

"""
    Strip whitespace:   ['A b. C d', 'A b. C d']
    Periods removed:    ['A b C d', 'A b C d']
    Apply lower:        ['a b c d', 'a b c d']
    Apply custom:       ['A B C D', 'A B C D']
    Apply regex:        ['x x x x', 'x x x x']
    Remove punctuations: ['A B C', '100 B C ']
"""