""" Regression Simulated Dataset (experience / salary)

Simulate the data for building a regression model.
Suppose, we have a survey among the employees of a company.

As a developer, often you have no access to survey data.
You need to simulate the data for building the regression model.
"""

import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

# Sample dataset
X, y = make_regression(
    n_samples = 100, 
    n_features = 1, # Employ years of experience
    n_informative = 1, 
    n_targets = 1,  # Employ's salary
    noise = 10, 
    coef = False, 
    random_state = 0
)

# Scale feature and target
X = np.interp(X, (X.min(), X.max()), (0, 20))           
y = np.interp(y, (y.min(), y.max()), (10000, 200000))   

# Fit a linear regression model
reg = LinearRegression().fit(X, y)


# Plot dataset points
plt.scatter(X, y, label='training data')
plt.title('Simultated dataset (Experience / Salary)')

# Plot the regression line 
x_line = np.linspace(np.min(X), np.max(X), 100)
y_line = reg.intercept_ + x_line * reg.coef_[0]

plt.plot(x_line, y_line, color='red', label='prediction')
plt.text(10, 25000, r'y = %0.2f + %0.2f x' % (reg.intercept_, reg.coef_[0]))
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.legend()
plt.show()


# Print data frame
df = pd.DataFrame(data={'Experience': X.flatten(), 'Salary': y})
print(df.head(10))
"""
	   Experience         Salary
	0    9.096218   95224.004179
	1   14.637429  132619.663538
	2   12.255808  123760.689176
	3    7.215160   98496.528556
	4    6.905628   80966.199869
	5   12.427999  138646.723320
	6    6.534503   62290.952298
	7   12.363590  129242.508929
	8   11.451010  132720.442525
	9    9.295277   93053.397973
"""