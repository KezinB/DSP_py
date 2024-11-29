import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Example data for class 1 and class 2
class1 = np.array([[1, 2, 1], [3, 3, 1], [4, 1, 1], [4, 2, 1],
                   [4, 3, 1], [4, 4, 1], [5, 1, 1], [5, 2, 1]])
class2 = np.array([[1.5, 7, 1], [2, 6, 1], [2, 8, 1], [3, 7, 1],
                   [3, 8, 1], [4, 7, 1], [4, 9, 1], [5, 5, 1]])

# Create a new figure for 3D plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot class1 points
ax.scatter(class1[:, 0], class1[:, 1], class1[:, 2], c='b', marker='o', label='Class1')

# Plot class2 points
ax.scatter(class2[:, 0], class2[:, 1], class2[:, 2], c='r', marker='^', label='Class2')

# Setting labels for the axes
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')

# Set the title and legend
ax.set_title('3D Scatter Plot')
ax.legend()

# Show the plot
plt.show()
