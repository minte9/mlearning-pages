## Decision Trees p71

Decision Trees model are used for `classification and regression` tasks.
They learn a `hierarchy if/else` questions, leading to a decision. 

In the machine learning, these questions are called `tests`.
For example, we can test if an animal `has feathers?` or `can fly?`.

Usually, data are `not in binary` form yes/no, but in a continuous form.
The tests used on `continuous data` are in the form of:
Is feature x `larger` than value x1?

### Boundaries

An `algorithm` that builds a decision tree will search over all posible tests 
and `find the one` that is most informative.

So, splitting the data `orizontally at x1` separates the points in class 0 and 1.
We can build a more accurate model by `repeating the process` of splitting.

This way will have a `decision boundaries` of tree with depth 2.
In the tree of decisions, `each node` containing a test.

This `recursive partitioning` of data is repeated until each region 
contains only a single target value (each `leaf` in decision tree).

### Prediction

A prediction of a new data point checks in `which partition` the point is, 
and then predict the `majority target` (or single target in case of pure leaves).

Usually, building a tree until `all the leaves` are pure leads to overfitting.
We can `stop the creation` of tree at some point (pre-prunning).
Or we can `remove or collapse` the nodes that are irelevant (prunning).

Decision tree model cannot generate new responses, outside of training data, 
as in linear models.


### References

> [Introduction to Machine Learning](https://www.amazon.com/gp/product/B01M0LNE8C) book  
> [Introduction to Machine Learning](https://github.com/amueller/introduction_to_ml_with_python/blob/master/02-supervised-learning.ipynb#Decision-trees) source code  
> [Learn and Remember](https://www.minte9.com/mlearning) minte9  