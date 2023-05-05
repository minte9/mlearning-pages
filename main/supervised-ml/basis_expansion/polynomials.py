""" Basis Expansion / Polynomial Model

Basis expansion is a technique used to transform a dataset by creating
new features from existing ones, in order to improve the accuracy of the model.

By expanding the set of input variables using basis functions, 
the model can capture nonlinear relationships between the inputs and the target variable, 
which might not be possible with linear models.

Even though the fifth-degree polynomial model has the lowest
SSR_training, it also has huge SSR_test, a good sign of overfitting.

Another sign of overfiting is by evaluating the coeficients (weights = sum of coeficients).
The higher the sum, the more the model tends to overfit.
"""

import numpy as np
import matplotlib.pyplot as plt

# Training datasets
X1 = [30, 46, 60, 65, 77, 95]  # area (m^2)
y1 = [31, 30, 80, 49, 70, 118] # price (10,000$)

# Test dataset
X2 = [17, 40, 55, 57, 70, 85]
y2 = [19, 50, 60, 32, 90, 110]

# N-degree polynomial
def pred_polinomial(degree, x_unknown, X1, y1):

    # Basis expansion & fit to training data
    p = np.poly1d(np.polyfit(X1, y1, degree))
    t = np.linspace(0, 100, 100)   

    # Plot train, test data and prediction line
    plt.figure(figsize=(6,4))
    plt.scatter(X1, y1, color='blue', label='Training set')
    plt.scatter(X2, y2, color='red', label='Test set')
    plt.plot(t, p(t), color='orange') # line

    # Evaluate the model (sum of residuals)
    SSR1 = sum((p(X1) - y1) ** 2).round()
    SSR2 = sum((p(X2) - y2) ** 2).round()

    # Evaluate the weight (sum of coeficients)
    weight = round(sum(abs(p.coef)))

    # Plot prediction
    xa = x_unknown
    ya = round(p(xa),2)
    plt.scatter(xa, ya, color='r', marker='x')
    plt.annotate(f'({xa}, {ya}, SSR1 = {SSR1}) SSR2 = {SSR2})', (xa+0.1, ya-10))

    plt.title(f'{degree}-degree polynomial')
    plt.legend(loc='best')
    plt.xlabel('Area (m^2)')
    plt.ylabel('Price (10,000$)')
    plt.xlim((0, 100))
    plt.ylim((0, 130))

    # Print prediction functions format
    xf1 = "{:.1f}x + {:.1f}"
    xf2 = "{:.1f}x^2 + " + xf1
    xf3 = "{:.1f}x^3 + " + xf2
    xf4 = "{:.1f}x^4 + " + xf3
    xf5 = "{:.1f}x^5 + " + xf4

    if degree == 1: 
        print(("p(x) = " + xf1).format(p[1], p[0]))
    elif degree == 2:
        print(("p(x) = " + xf2).format(p[2], p[1], p[0]))
    elif degree == 3:
        print(("p(x) = " + xf3).format(p[3], p[2], p[1], p[0]))
    elif degree == 4:
        print(("p(x) = " + xf4).format(p[4], p[3], p[2], p[1], p[0]))
    elif degree == 5:
        print(("p(x) = " + xf5).format(p[5], p[4], p[3], p[2], p[1], p[0]))

    print('SSR1 =', SSR1, ' / ', 'SSR2 =', SSR2)
    print('weight = ', weight, '\n')
    return


# Plot prediction
x_unknown = 50
pred_polinomial(1, x_unknown, X1, y1)
pred_polinomial(2, x_unknown, X1, y1) 
pred_polinomial(3, x_unknown, X1, y1) 
pred_polinomial(4, x_unknown, X1, y1) 
pred_polinomial(5, x_unknown, X1, y1) 
plt.show()

"""
    p(x) = 1.3x + -18.0
    SSR1 = 1248.0  /  SSR2 = 1681.0
    weight =  19 

    p(x) = 0.0x^2 + -0.5x + 31.9
    SSR1 = 995.0  /  SSR2 = 1530.0
    weight =  32 

    p(x) = 0.0x^3 + -0.0x^2 + 3.0x + -29.5
    SSR1 = 967.0  /  SSR2 = 1671.0
    weight =  33 

    p(x) = 0.0x^4 + -0.0x^3 + 1.8x^2 + -66.5x + 876.9
    SSR1 = 651.0  /  SSR2 = 29011.0
    weight =  945 

    p(x) = -0.0x^5 + 0.0x^4 + -1.1x^3 + 66.8x^2 + -1866.2x + 19915.1
    SSR1 = 0.0  /  SSR2 = 6719065.0
    weight =  21849 
"""