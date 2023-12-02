import numpy as np
import matplotlib.pyplot as plt

# Training datasets
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

class LinearRegression:

    def __init__(self):
        self.coef_ = []
        self.intercept_ = 0

    def fit(self, X_train, Y_train, learning_rate=0.01, num_iterations=1000):
        
        x = X_train
        y = Y_train
        m = 0 
        b = 0
        for i in range(num_iterations):
            y_pred = m*x + b
            error = y - y_pred

            m_derivative = -(2/len(x)) * sum(x * error)
            b_derivative = -(2/len(x)) * sum(error)

            m -= learning_rate * m_derivative
            b -= learning_rate * b_derivative

        obj = LinearRegression()
        obj.coef_.append(m)
        obj.intercept_ = b
        
        return obj

# Learn a prediction function
r = LinearRegression().fit(x, y)
m = r.coef_[0].round(1)
b = r.intercept_.round(1)

# Prediction
x1 = 3
y1 = m*x1 + b

# Output
print('Best line:', f"y = {m}x + {b}")
print('Prediction for x=3:', f"y = {y1:.1f}")

m = round(m, 1)
b = round(b, 1)

fig, ax = plt.subplots()
plt.ylim(0, 10)
plt.xlim(0, 10)

ax.plot(x,  y,  'x', color='g', label='Training data')
ax.plot(x, m*x + b,  label=f'h(x) = {m} + {b}x')
ax.plot(x1, y1, 'o', color='r', label=f'h({x1}) = {y1}') # Draw unknown point
plt.legend()
plt.show()