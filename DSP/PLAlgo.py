import numpy as np
import matplotlib.pyplot as plt

# Data for class 1 and class 2
class1 = np.array([[1, 2, 1], [3, 3, 1], [4, 1, 1], [4, 2, 1],
                   [4, 3, 1], [4, 4, 1], [5, 1, 1], [5, 2, 1]])
class2 = np.array([[1.5, 7, 1], [2, 6, 1], [2, 8, 1], [3, 7, 1],
                   [3, 8, 1], [4, 7, 1], [4, 9, 1], [5, 5, 1]])
norm_class2 = -class2

# Initial settings for perceptron without margin
k = 0
a = np.array([0, 0, 0])
eta = 1
Y = np.vstack((class1, norm_class2))

# Training loop for perceptron without margin
while k < 500:
    prev_a = a.copy()
    for i in range(len(Y)):
        temp = Y[i, :]
        ax = np.dot(a, temp)
        if ax <= 0:
            a = a + eta * temp
    if np.array_equal(prev_a, a):
        break
    k += 1

# Decision boundary calculation
x1 = np.arange(0, 6.1, 0.1)
x2 = (-a[0] / a[1]) * x1 - a[2] / a[1]

# Plotting without margin
plt.scatter(class1[:, 0], class1[:, 1])
plt.scatter(class2[:, 0], class2[:, 1])
plt.plot(x1, x2)
plt.grid(True)
plt.title('Perceptron without Additional Margin')
plt.legend(['Separating Plane', 'Class1', 'Class2'])
plt.show()

# Data for class 1 and class 2 (again for margin perceptron)
class1 = np.array([[1, 2, 1], [3, 3, 1], [4, 1, 1], [4, 2, 1],
                   [4, 3, 1], [4, 4, 1], [5, 1, 1], [5, 2, 1]])
class2 = np.array([[1.5, 7, 1], [2, 6, 1], [2, 8, 1], [3, 7, 1],
                   [3, 8, 1], [4, 7, 1], [4, 9, 1], [5, 5, 1]])
norm_class2 = -class2

# Initial settings for perceptron with margin
k = 0
b = 2.6
a = np.array([0, 0, 0])
eta = 1
Y = np.vstack((class1, norm_class2))

# Training loop for perceptron with margin
while k < 500:
    prev_a = a.copy()
    for i in range(len(Y)):
        temp = Y[i, :]
        ax = np.dot(a, temp)
        if ax <= b:
            dist_point = b - ax
            a = a + (eta * (dist_point * temp) / np.linalg.norm(temp)**2)
    if np.array_equal(prev_a, a):
        break
    k += 1

# Decision boundary calculation
x1 = np.arange(0, 6.1, 0.1)
x2 = (-a[0] / a[1]) * x1 - a[2] / a[1]

# Plotting with margin
plt.scatter(class1[:, 0], class1[:, 1])
plt.scatter(class2[:, 0], class2[:, 1])
plt.plot(x1, x2)
plt.grid(True)
plt.title('Perceptron with Additional Margin')
plt.legend(['Separating Plane', 'Class1', 'Class2'])
plt.show()
