import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 3D Data for class 1 and class 2
class1 = np.array([[1, 2, 1], [3, 3, 1], [4, 1, 1], [4, 2, 1],
                   [4, 3, 1], [4, 4, 1], [5, 1, 1], [5, 2, 1]])
class2 = np.array([[1.5, 7, 2], [2, 6, 2], [2, 8, 2], [3, 7, 2],
                   [3, 8, 2], [4, 7, 2], [4, 9, 2], [5, 5, 2]])
norm_class2 = -class2

# Initial settings for perceptron without margin
k = 0
a = np.array([0, 0, 0, 0])
eta = 1
Y = np.vstack((class1, norm_class2))

# Training loop for perceptron without margin
while k < 500:
    prev_a = a.copy()
    for i in range(len(Y)):
        temp = np.append(Y[i, :], 1)
        ax = np.dot(a, temp)
        if ax <= 0:
            a = a + eta * temp
    if np.array_equal(prev_a, a):
        break
    k += 1

# Decision boundary calculation
x1 = np.arange(0, 6.1, 0.1)
x2 = np.arange(0, 10, 0.1)
X1, X2 = np.meshgrid(x1, x2)
x3 = (-a[0] * X1 - a[1] * X2 - a[3]) / a[2]

# 3D Plotting without margin
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(class1[:, 0], class1[:, 1], class1[:, 2], c='b', marker='o', label='Class1')
ax.scatter(class2[:, 0], class2[:, 1], class2[:, 2], c='r', marker='^', label='Class2')
ax.plot_surface(X1, X2, x3, alpha=0.3)
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
ax.set_title('Perceptron without Additional Margin')
ax.legend()

plt.show()

# Initial settings for perceptron with margin
k = 0
b = 2.6
a = np.array([0, 0, 0, 0])
eta = 1
Y = np.vstack((class1, norm_class2))

# Training loop for perceptron with margin
while k < 500:
    prev_a = a.copy()
    for i in range(len(Y)):
        temp = np.append(Y[i, :], 1)
        ax = np.dot(a, temp)
        if ax <= b:
            dist_point = b - ax
            a = a + (eta * (dist_point * temp) / np.linalg.norm(temp)**2)
    if np.array_equal(prev_a, a):
        break
    k += 1

# Decision boundary calculation
x1 = np.arange(0, 6.1, 0.1)
x2 = np.arange(0, 10, 0.1)
X1, X2 = np.meshgrid(x1, x2)
x3 = (-a[0] * X1 - a[1] * X2 - a[3]) / a[2]

# 3D Plotting with margin
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(class1[:, 0], class1[:, 1], class1[:, 2], c='b', marker='o', label='Class1')
ax.scatter(class2[:, 0], class2[:, 1], class2[:, 2], c='r', marker='^', label='Class2')
ax.plot_surface(X1, X2, x3, alpha=0.3)
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
ax.set_title('Perceptron with Additional Margin')
ax.legend()

plt.show()
