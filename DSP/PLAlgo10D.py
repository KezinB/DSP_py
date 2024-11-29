import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# Generate random 10D data for two classes
np.random.seed(0)
class1 = np.random.rand(100, 10) * 10
class2 = np.random.rand(100, 10) * 10 + 10
print(class1)
print("class2 : ")
print(class2)
# Combine the data for PCA
data = np.vstack((class1, class2))
labels = np.array([0]*100 + [1]*100)

# Apply PCA to reduce to 3 dimensions
pca = PCA(n_components=3)
reduced_data = pca.fit_transform(data)

# Split the reduced data back into classes
class1_reduced = reduced_data[labels == 0]
class2_reduced = reduced_data[labels == 1]

# Plotting the reduced data in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(class1_reduced[:, 0], class1_reduced[:, 1], class1_reduced[:, 2], c='b', marker='o', label='Class1')
ax.scatter(class2_reduced[:, 0], class2_reduced[:, 1], class2_reduced[:, 2], c='r', marker='^', label='Class2')

# Setting labels
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.set_title('PCA of 10D Data to 3D')
ax.legend()

plt.show()
