""" Cleaning Text

Most basic cleaning can be done with core string operations
In the real world we will most likely define a custom cleaning function.

Translate is a Python method popular due to its blazing speed.
To remove all punctuation we translate the P chars to None, effectively 
removing them, which is far faster than alternatives.
"""

import string
import re
import sys
import unicodedata

A = ["  A!!! b. C.... d  ", "  100% A! #b ?!  "]

def clear_text(A):
    A = [s.strip() for s in A]              # strip whitespaces
    A = [s.replace(".", "") for s in A]     # remove periods
    A = [s.lower() for s in A]              # apply lower
    return A

def apply_function(s: str) -> str: 
    return s.upper()

def apply_regex(s: str) -> str:
    return re.sub(r"[a-zA-Z]", "x", s)

def remove_punctuation(A):
    P = dict.fromkeys( # first (fromkeys)
        i for i in range(sys.maxunicode) 
            if unicodedata.category(chr(i)).startswith('P')
    )
    P = dict() # second (loop)
    for i in range(sys.maxunicode):
        if unicodedata.category(chr(i)).startswith('P'):
            P[i] = None
    A = [s.translate(P) for s in A]
    return A

# -------------------------------------------------------

# Output
print("Input text:", A)

A = clear_text(A)
print("Clear text:", A)

A = [apply_function(s) for s in A]
print("Upper transform:", A)

A = [apply_regex(s) for s in A]
print("Regex transform:", A)

A = remove_punctuation(A)
print("Remove punctuation:", A)

"""
    Input text: ['  A!!! b. C.... d  ', '  100% A! #b ?!  ']
    Clear text: ['a!!! b c d', '100% a! #b ?!']
    Upper transform: ['A!!! B C D', '100% A! #B ?!']
    Regex transform: ['x!!! x x x', '100% x! #x ?!']
    Remove punctuation: ['x x x x', '100 x x ']
"""