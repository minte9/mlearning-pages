""" Simulated dataset (experience / salary)

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
X, y = make_regression(n_samples = 100, n_features = 1, n_informative = 1, n_targets = 1,
    noise = 10, coef = False, random_state = 0
)
print(X[:1]) # [[-0.35955316]]
print(y[:1]) # [-19.95588561]


# Scale
X = np.interp(X, (X.min(), X.max()), (0, 20))           # years of experience
y = np.interp(y, (y.min(), y.max()), (10000, 200000))   # salary

# Dataframe
df = pd.DataFrame(data={'Experience': X.flatten(), 'Salary': y})
print(df.head(2))
    #  Experience         Salary
    #    9.096218   95224.004179
    #   14.637429  132619.663538

# Plot dataset points
plt.scatter(X, y, label='training data')
plt.title('Simultated dataset (Experience / Salary)')


# Fit a linear regression model
reg = LinearRegression().fit(X, y)

# Plot the regression line 
x_line = np.linspace(np.min(X), np.max(X), 100)
y_line = reg.intercept_ + x_line * reg.coef_[0]
plt.plot(x_line, y_line, color='red', label='prediction')
plt.text(10, 25000, r'y = %0.2f + %0.2f x' % (reg.intercept_, reg.coef_[0]))
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.legend()
plt.show()