import numpy as np
def eigen_decomposition(A, max_iterations, tolerance):
    n = A.shape[0]
    eigenvectors = np.eye(n)
    eigenvalues = np.zeros(n)
    for k in range(n):
        x = np.random.randn(n)
        x = x / np.linalg.norm(x)
        for iteration in range(max_iterations):
            y = A @ x
            lambda_ = x @ y
            x = y / np.linalg.norm(y)
            if np.linalg.norm(A @ x - lambda_ * x) < tolerance:
                break
        eigenvalues[k] = lambda_
        eigenvectors[:, k] = x
        A = A - lambda_ * np.outer(x, x)
    return eigenvalues, eigenvectors

A = np.array(eval(input('Enter the matrix A in Python format: ')))
max_iterations = 1000
tolerance = 1e-6
eigenvalues, eigenvectors = eigen_decomposition(A, max_iterations, tolerance)
print('Eigenvectors:')
print(eigenvectors)
print('Eigenvalues:')
print(eigenvalues)