import numpy as np
import matplotlib.pyplot as plt

# Define variables
ones = np.ones(3)  # Create an array of ones
A = np.array([[0, 1], [1, 1], [2, 1]])  # Matrix A
xfeature = A.T[0]  # Extract the first column (x-feature)
squaredfeature = A.T[0] ** 2  # Square of the first column

# Vector b and reshaping
b = np.array([1, 2, 0], ndmin=2).T  # Ensure it's a 2D column vector
b = b.reshape(3)  # Reshape it back to a 1D array

# Create feature matrix by stacking arrays
features = np.concatenate((np.vstack(ones), np.vstack(xfeature), 
                           np.vstack(squaredfeature)), axis=1)

# Create a copy of features for later modification
featuresc = features.copy()

print("Features Matrix:\n", features)

# Calculate the determinant of the original feature matrix
m_det = np.linalg.det(features)
print("Determinant of Feature Matrix:", m_det)

# Initialize a list to store determinants
determinants = []

# Loop over columns, replacing each with vector b
for i in range(3):
    featuresc[:, i] = b  # Replace column 'i' with vector 'b'
    print(f"Modified Features Matrix (Column {i} replaced by b):\n", featuresc)
    
    # Calculate the determinant of the modified matrix
    det = np.linalg.det(featuresc)
    determinants.append(det)
    
    # Reset featuresc to the original matrix for the next iteration
    featuresc = features.copy()

# Calculate the final coefficients by dividing determinants by m_det
determinants = np.array(determinants) / m_det
print("Coefficients:", determinants)

plt.title("Least Squares Fit to a Quadratic Polynomial")
# Plot the data points
plt.scatter(A.T[0], b, label="Data Points")

# Generate values for the fitted curve
u = np.linspace(0, 3, 100)
plt.plot(u, u**2 * determinants[2] + u * determinants[1] + determinants[0], label="Fitted Curve")

# Use numpy's polyfit to fit a polynomial and plot it
p2 = np.polyfit(A.T[0], b, 2)
plt.plot(u, np.polyval(p2, u), 'r--', label="Polyfit Curve (Degree 2)")

# Add legend and show the plot
plt.legend()
plt.show()
