import numpy as np

R = int(input("Enter the number of rows: "))
C = int(input("Enter the number of columns: "))

# Initialize the matrix
matrix = [[0 for c in range(C)] for r in range(R)]
for r in range(R):
    for c in range(C):
        matrix[r][c] = int(input(f"Enter element at row {r + 1}, column {c + 1}: "))

matrix_array = np.array(matrix)
print("The given matrix is:\n", matrix_array)

# Loop to reduce the matrix to RREF
lead = 0
for r in range(R):
    if lead >= C:
        break
    pivot = matrix_array[r][lead]
    if pivot == 0:
        for i in range(r + 1, R):
            if matrix_array[i][lead] != 0:
                matrix_array[[r, i]] = matrix_array[[i, r]]
                pivot = matrix_array[r][lead]
                break
    if pivot != 0:
        matrix_array[r] = matrix_array[r] / pivot
        for i in range(R):
            if i != r:
                factor = matrix_array[i][lead]
                matrix_array[i] -= factor * matrix_array[r]
        lead += 1

print("\nReduced Row Echelon Form (RREF):\n", matrix_array)

# Calculate the rank
count = 0
for i in range(R):
    if np.all(matrix_array[i, :] == 0):
        count += 1
rank = R - count
print("Rank of matrix =", rank)
