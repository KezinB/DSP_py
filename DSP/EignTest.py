import numpy as np
A = np.array(eval(input('Enter the matrix A in Python format: ')))
n = A.shape[0]
print(A)
I = np.eye(n)
print(I)
mat = A-I
print(mat)