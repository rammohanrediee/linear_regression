import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("test.csv")
x = data['x'].values
y = data['y'].values

# Feature Scaling
x = (x - np.mean(x)) / np.std(x)  # Standardization

alpha = 0.01  # Reduced Learning Rate
w = 0.0
b = 0.0
n_iter = 1000
m = len(x)

for i in range(n_iter):
    y_pred = w * x + b
    error = y_pred - y
    w_grad = (2 / m) * np.sum(error * x)
    b_grad = (2 / m) * np.sum(error)

    w -= alpha * w_grad
    b -= alpha * b_grad

sorted_indices = np.argsort(x)
x_sorted = x[sorted_indices]
y_test = w * x_sorted + b

plt.scatter(x, y, color="blue", label="Actual Data")
plt.plot(x_sorted, y_test, color="red", label="Regression Line")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Linear Regression Fit")
plt.show()
