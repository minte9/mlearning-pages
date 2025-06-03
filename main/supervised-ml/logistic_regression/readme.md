## Logistic Regression

### Binary
 p259
Despite the name, logistic regression is actually a widely used classification technique.  
The logistic function is an S-shaped curve that maps any real-value number to 0 and 1.  

### Cost Function

Logistic Regression is similar to Linear Regression but the model uses a more complex cost function.  
Linear Regression:   hΘ(x) = β₀ + β₁X  
Logistic Regression: hΘ(x) = 1/(1 + e^-(β₀ + β₁X))  

### Output

The output of the logistic regression is a probability value between 0 and 1, which represents the  
likelihood of belonging to a specific class.

### Scalling
 p260
As our data have different values, and even different measurement units,  
we use scalling in order to compare them.  

### References

[Introduction to Logistic Regression](https://towardsdatascience.com/introduction-to-logistic-regression-66248243c148)
[Develop a Logistic Regression](https://blog.devgenius.io/develop-a-logistic-regression-machine-learning-model-64d2be403ba3)