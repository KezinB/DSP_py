import numpy as np
import matplotlib.pyplot as plt

# Define variables
A = np.array([[0, 1], [1, 1], [2, 1]])  # Matrix A
xfeature = A.T[0]  # Extract the first column (x-feature)
squaredfeature = A.T[0] ** 2  # Square of the first column

# Vector b and reshaping
b = np.array([1, 2, 0], ndmin=2).T  # Ensure it's a 2D column vector
b = b.reshape(3)  # Reshape it back to a 1D array

# Add noise to b
noise = np.random.normal(0, 0.1, b.shape)  # Mean 0, standard deviation 0.1
b_noisy = b + noise  # Add noise to the original data

print("Original Data Points (b):", b)
print("Noisy Data Points (b_noisy):", b_noisy)

# Fitting the noisy data to a quadratic polynomial using np.polyfit
coefficients = np.polyfit(xfeature, b_noisy, 2)

print("Coefficients:", coefficients)

plt.title("Least Squares Fit to a Quadratic Polynomial with Noise")
# Plot the noisy data points
plt.scatter(xfeature, b_noisy, label="Data Points with Noise")

# Generate values for the fitted curve
u = np.linspace(0, 2, 100)
plt.plot(u, u**2 * coefficients[0] + u * coefficients[1] + coefficients[2], label="Fitted Curve")

# Use numpy's polyfit to fit a polynomial and plot it
p2 = np.polyfit(xfeature, b_noisy, 2)
plt.plot(u, np.polyval(p2, u), 'r--', label="Polyfit Curve (Degree 2)")

# Add legend and show the plot
plt.legend()
plt.show()
