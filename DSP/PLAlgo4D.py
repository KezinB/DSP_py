import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Generating 4D data
np.random.seed(0)
class1 = np.random.rand(100, 4) * 10
class2 = np.random.rand(100, 4) * 10 + 10

# Plotting 3D and using color for the fourth dimension
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for class1
sc1 = ax.scatter(class1[:, 0], class1[:, 1], class1[:, 2], c=class1[:, 3], cmap='viridis', label='Class1')

# Scatter plot for class2
sc2 = ax.scatter(class2[:, 0], class2[:, 1], class2[:, 2], c=class2[:, 3], cmap='plasma', label='Class2')

# Adding colorbars to represent the fourth dimension
cbar1 = plt.colorbar(sc1, ax=ax, shrink=0.5, aspect=10)
cbar1.set_label('Class1 - Fourth Dimension')
cbar2 = plt.colorbar(sc2, ax=ax, shrink=0.5, aspect=10)
cbar2.set_label('Class2 - Fourth Dimension')

# Setting labels
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
ax.set_title('4D Data Plot with Color representing the Fourth Dimension')
ax.legend()

plt.show()
