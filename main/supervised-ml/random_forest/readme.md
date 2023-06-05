## Random Forest

### Random
 p11  
Decision Tree is a step by step process in order to decide which category an item belongs to.  
Random Forest is a collection of Decistion Tree generated using a random subset of data.  

### Threshold
 p26  
Decistion Tree picks a criteria and a thresold.  
Criteria specifies where to split, for instance length or with.  
The thresold specifies what value of the criteria to split.  

### The split
 p26
In order to make a split a human will simply draw a line.  
A computer decisiont tree classifier would need more steps to do it.  
But if we have a relationship between different criteria it's easier. 

For example, y = 9.5 - x for red dots and y = 10.5 - x for blue dots.  
The Random Forest can make the split in a single line.  
Anything with y bellow 10 is red, and anything with y above 10 is blue.  

### Overfitting
 p31  
By default, the computer never stops.  
It continues until every single piece of data is split. 
This leads to overfitting. 

The purpose of decision tree classifier is to be able to tell   
which type is an unknown item and overfitting doesn't help with that.  
The new item will be a bit different from what we know from data.  

### Limit overfitting
 p32  
One way of limiting overfiting is by limit the number of splits.  
The other way is to split only if the brach has a minimum items in it.  

### Random Forests
 p38  
Random Forests try to fix this overfitting by using multiple decision trees  
that are slightly different and averaging the results.  
Using the same data over and over for generating Decistion Trees will result  
in many copies of the same tree.  
Random Forests are using subsets of the dataset, randomly selected.  

### Bootstrapping
 p45  
The generated trees have unique set of data.  
The data is selected randomly for the original dataset, with replacement.  
Each one of the tree has the same size as the original data.  

dataset: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10  
subset1: 1, 2, 2, 4, 5, 7, 7, 7, 9, 10  
subset2: 2, 3, 3, 5, 6, 6, 7, 7, 8, 9  

### Out of bag

Random Forest set aside some data, using on average only 63.2% of the dataset.  
Insteed of test data we can use this out of bag for checking the accuracy of the model.  

### References

[Machine Learning With Random Forests And Decision Trees](https://www.amazon.com/gp/product/B01JBL8YVK)
[Understanding Random Forest](https://towardsdatascience.com/understanding-random-forest-58381e0602d2)
[Random Forest Example](https://www.analyticsvidhya.com/blog/2021/06/understanding-random-forest/)
[Radom Forest Python](https://vitalflux.com/random-forest-classifier-python-code-example/)
[What is Random Forest](https://www.youtube.com/watch?v=gkXX4h3qYm4&ab_channel=IBMTechnology)