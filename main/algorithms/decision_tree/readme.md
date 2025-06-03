## Decision Trees - Algorithm

### Entropy

Entropy is a measure of how `disordered` a collection of data is.  
P1 and p2 represents the `probability of ocurrance` of that sample in the given data.  
As an example, for 14 data samples (`9 positive`, 5 negative):  

    H = -(p1*log(p1) + p2*log(p2))  
    H = -(9/14)log(9/14) - (5/14)log(5/14) = 0.940  

### Information gain

The amount of information gained from a sample is known as `information gain`.  
We can have the attribute `wind` (9 positive, 5 negative).  

The values for that attribute can be `week` or`strong`. 
For example, in case of week (6 yes, 2 no) and strong (3 yes, 3 no),  
we can compute information gain as: 

    IG = H - (8/14)H_week - (6/14)H_strong
    IG = 0.940 - (8/14)0.811 - (6/14)1.00 - 0.048

### Decision Tree

Decision tree is a tree developed based on an `algorithm decisions`.  
The algorithm uses `the features` for learning and generate a tree structure.  
The decision tree can then be used to make `predictions`.  

A decision tree is a `supervised ML algorithm`, for classification and regression models   
that generates a tree structure that can be used later for predictions.  
    Node - a feature (or attribute)
    Branch - a decistion (rule or test)
    Leaf - an outcome (categorical or continuous)

### Algorithm

1. Calculate `entropy` for dataset  
2. For each `feature`:  
    Calculate `entropy for all` categorical values  
    Calculate `information gain` for the feature  
3. Find the feture with `maximum` information gain  
4. Repeat  

#

### References

> [Decision Tree Types](https://www.knowledgehut.com/blog/data-science/classification-and-regression-trees-in-machine-learning)  Entropy  
> [Information Gain](https://www.featureranking.com/tutorials/machine-learning-tutorials/information-gain-computation)  
> [Entropy Formula](https://www.mathsisfun.com/physics/entropy.html)  mathisfun  
> [Entropy Formula](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html)  python  
> [PlayTennis Dataset](https://www.kaggle.com/code/sdk1810/decision-tree-for-playtennis)  kaggle  
> [ID3 Algorithm Implementation](https://www.kaggle.com/code/smsmibrahim/decision-tree-id3-implementation-using-play-tennis/notebook)  notebook  
> [ID3 Algorithm](https://www.enjoyalgorithms.com/blog/decision-tree-algorithm-in-ml)  enjoy  
> [ID3 Algorithm](https://automaticaddison.com/iterative-dichotomiser-3-id3-algorithm-from-scratch/)  from scrach  
> [Learn and Remember](https://www.minte9.com/mlearning/algorithms-decision-tree-1474) minte9  