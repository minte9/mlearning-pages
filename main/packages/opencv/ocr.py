"""Image to text
Open Cv (Open Source Computer Vision Library) used to load images.
Tesseract used for optical character recognition.
"""
import os
DIR = os.path.dirname(os.path.realpath(__file__))

import cv2, pytesseract
img = cv2.imread(DIR + '/files/01.png')
text = pytesseract.image_to_string(img)

print(text)

"""
I've also done a lot of testing since LiveJournal.
Once I started working with other people
especially. And once I realized that code I write
never fucking goes away and I'm going to be a
maintainer for life. I get comments about blog
posts that are almost 10 years old. “Hey, I found
this code. I found a bug,” and I'm suddenly
maintaining code.
"""