import numpy as np
import matplotlib.pyplot as plt

# Define classes
class1 = np.array([[1, 2], [3, 3], [4, 1], [4, 2], [6, 7], [9, 1], [7, 5], [1, 1], [3, 5]])
class2 = np.array([[1.5, 7], [2, 6], [2, 8], [3, 7], [1, 2], [3, 3], [4, 1], [4, 2], [4, 3]])

# Labels
labels_class1 = np.ones(class1.shape[0])
labels_class2 = -1 * np.ones(class2.shape[0])

# Combine data and labels
X = np.vstack((class1, class2))
y = np.hstack((labels_class1, labels_class2))

# Initialize weights and bias
w = np.zeros(X.shape[1])
b = 0
eta = 1
max_iter = 500

# Perceptron learning algorithm
for _ in range(max_iter):
    for i in range(X.shape[0]):
        if y[i] * (np.dot(w, X[i]) + b) <= 0:
            w = w + eta * y[i] * X[i]
            b = b + eta * y[i]

# Plot decision boundary
x1 = np.linspace(min(X[:, 0]), max(X[:, 0]), 10)
x2 = -(w[0] * x1 + b) / w[1]

plt.scatter(class1[:, 0], class1[:, 1], label='Class 1', edgecolor='black')
plt.scatter(class2[:, 0], class2[:, 1], label='Class 2', edgecolor='black')
plt.plot(x1, x2, label='Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.title('Perceptron Decision Boundary')
plt.show()
