import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Generate random data from a normal distribution
data = np.random.randn(1000)  # 1000 samples, mean = 0, std = 1

# Plot the histogram, normalized to form a probability density
plt.hist(data, bins=30, density=True, alpha=0.6, color='b', edgecolor='black')

# Generate the x values for the normal distribution curve
x = np.linspace(min(data), max(data), 100)
y = norm.pdf(x, 0, 1)  # PDF of standard normal distribution (mean=0, std=1)

# Plot the normal distribution curve
plt.plot(x, y, 'r', linewidth=2)

# Add labels and title
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.title('Histogram of Normally Distributed Data with PDF Overlay')

# Show the plot
plt.show()
