# Medium Article: https://larswaechter.medium.com/a-decent-introduction-to-gradient-descent-in-python-846be2e41592

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# The model
def predict(X, a, b):
    return X * a + b # f(x) = ax + b

# The derivatives
def gradient(X, Y, a, b):
    da = 2 * np.mean((predict(X, a, b) - Y) * X)
    db = 2 * np.mean((predict(X, a, b) - Y) * 1)
    return (da, db)

# Train the model
def train(X, Y, iterations, lr):
    a = b = 0

    # Gradient Descents iterations
    for i in range(iterations):
        da, db = gradient(X, Y, a, b)
        a -= da * lr
        b -= db * lr
    return a, b

X = np.array([15, 18, 20, 21, 23, 25, 27, 28, 29, 30, 32, 34, 35, 36])
Y = np.array([23, 74, 65, 82, 135, 321, 440, 400, 290, 620, 630, 610, 560, 568])

# X = np.array([30, 46, 60, 65, 77, 95])
# Y = np.array([31, 30, 80, 49, 70, 118])

a, b = train(X, Y, iterations=20000, lr=0.001)

print("\na=%.8f; b=%.8f" % (a, b))
print("Prediction: x=%d => y=%.2f" % (33, predict(33, a, b)))