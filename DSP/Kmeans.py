import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Generate random data
N = 100000
K = 3
C = np.random.rand(K, 2) * 10 - 5
X = np.zeros((N, 2))
for i in range(N):
    k = np.random.randint(K)
    X[i] = C[k] + np.random.randn(2)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=K)
kmeans.fit(X)
idx = kmeans.labels_
C = kmeans.cluster_centers_

# Plot data points and cluster centers
plt.scatter(X[:, 0], X[:, 1], c=idx, cmap='viridis', s=15)
plt.scatter(C[:, 0], C[:, 1], c='black', marker='x', s=100, label='Cluster Centers')

# Create decision boundaries
x1 = np.arange(X[:, 0].min(), X[:, 0].max(), 0.01)
x2 = np.arange(X[:, 1].min(), X[:, 1].max(), 0.01)
X1, X2 = np.meshgrid(x1, x2)
X_grid = np.c_[X1.ravel(), X2.ravel()]
idx_grid = kmeans.predict(X_grid)
decision_boundary = idx_grid.reshape(X1.shape)

# Plot decision boundaries
plt.contour(X1, X2, decision_boundary, colors='black', linewidths=1)
plt.title('K-Means Clustering with Decision Region Plot')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
