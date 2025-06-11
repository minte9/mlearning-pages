"""Image to text and get only highlighted text.
Converting to HSV makes color selection easier.
"""
import os
import cv2, pytesseract
import numpy as np
DIR = os.path.dirname(os.path.realpath(__file__))

def imread_highlighted(img):

    # Convert BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 

    # Range of yellow color in HSV
    lower = np.array([22, 93, 0]) 
    upper = np.array([45, 255, 255])

    # Mask to get only yellow colors
    mask = cv2.inRange(hsv, lower, upper)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(img, img, mask= mask)

    # Invert the mask to get black letters on white background
    res2 = cv2.bitwise_not(mask)

    # Display images
    if False:
        cv2.imshow("img", res)
        cv2.imshow("img2", res2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return res2


img = cv2.imread(DIR + '/files/01.png')
img2 = imread_highlighted(img)

text = pytesseract.image_to_string(img).strip()
highlighted = pytesseract.image_to_string(img2).strip()
replaced = text.replace(highlighted, '<i>%s</i>' % highlighted)

print('Text: \n' + text, '\n')
print('Highlighted: \n' + highlighted, '\n')
print('Replaced: \n' + replaced, '\n')

"""
Text: 
I've also done a lot of testing since LiveJournal.
Once I started working with other people
especially. And once I realized that code I write
never fucking goes away and I'm going to be a
maintainer for life. I get comments about blog
posts that are almost 10 years old. “Hey, I found
this code. I found a bug,” and I'm suddenly
maintaining code. 

Highlighted: 
Once I started working with other people
especially. And once I realized that code I write
never fucking goes away and I'm going to be a
maintainer for life. 

Replaced: 
I've also done a lot of testing since LiveJournal.
<i>Once I started working with other people
especially. And once I realized that code I write
never fucking goes away and I'm going to be a
maintainer for life.</i> I get comments about blog
posts that are almost 10 years old. “Hey, I found
this code. I found a bug,” and I'm suddenly
maintaining code. 
"""