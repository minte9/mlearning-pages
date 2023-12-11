"""Image to text - highlight mask
Some character might not be correctly detected.
We can reduse noise, remove new lines, use string slices, 
check punctuation, etc ...
"""
import os, sys
import cv2, pytesseract, numpy as np
import re
DIR = os.path.dirname(os.path.realpath(__file__))

def imread_highlighted(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([22, 93, 0])
    upper = np.array([45, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    res = cv2.bitwise_and(img, img, mask= mask)
    res2 = cv2.bitwise_not(mask)
    res2 = cv2.GaussianBlur(res2, (3,3), 0) # reduse noise

    if False:
        cv2.imshow("img", res)
        cv2.imshow("img2", res2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return res2

def highlighted_replaced(img, img2):
    text = pytesseract.image_to_string(img).strip()
    highlighted = pytesseract.image_to_string(img2).strip()

    text = '\n'.join(text.split('\n\n')) # remove double new lines
    highlighted = '\n'.join(highlighted.split('\n\n'))

    pattern = re.compile(r',$') # wrong , end of propostion
    highlighted = pattern.sub(r'', highlighted) # remove

    start = highlighted[0:10].strip() # start of highlighted text
    end = highlighted[-10:].strip()

    replaced = text.replace(start, '<i>%s' % start)
    replaced = replaced.replace(end, '%s</i>' % end)

    if False:
        print('Text: \n' + text, '\n')
        print('Highlighted: \n' + highlighted, '\n')
        print('Start: \n' + start, '\n')
        print('End: \n' + end, '\n')
        print('Replaced: \n' + replaced, '\n')

    try:
        assert (start in text), '<%s> not in text' % start
        assert (end in text), '<%s> not in text' % end
    except AssertionError as e:
        print('AssertionError:\n', e)

    return replaced

for root, dirs, files in os.walk(DIR + '/files/'):
    for file in files:
        img = cv2.imread(DIR + '/files/' + file)
        img2 = imread_highlighted(img)
        replaced = highlighted_replaced(img, img2)
        print(replaced, '\n')

"""
where it was writing some big file. <i>We took really
good advantage of multithreading in Java, which
was less painful than I had expected it to be. It was
just really pleasant to work on.</i> From the API we
had designed we saw all these directions it could
grow. 

I've also done a lot of testing since LiveJournal.
<i>Once I started working with other people
especially. And once I realized that code I write
never fucking goes away and I'm going to be a
maintainer for life.</i> I get comments about blog
posts that are almost 10 years old. “Hey, I found
this code. I found a bug,” and I'm suddenly
maintaining code. 

Your competitor's six-month 1.0 has crap code and
<i>they're going to have to rewrite it in two years but</i>,
guess what: they can rewrite it because you don't
have ajob anymore. 
"""