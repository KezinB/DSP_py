import numpy as np

def is_rref(matrix):
    matrix = np.array(matrix)
    rows, cols = matrix.shape
    lead = 0
    
    for r in range(rows):
        if lead >= cols:
            return True
        i = r
        while matrix[i][lead] == 0:
            i += 1
            if i == rows:
                i = r
                lead += 1
                if cols == lead:
                    return True
        if matrix[i][lead] != 1:
            return False
        for j in range(rows):
            if j != i and matrix[j][lead] != 0:
                return False
        lead += 1
    
    # Check for zero rows at the bottom
    for i in range(rows):
        if np.all(matrix[i, :] == 0):
            continue
        first_non_zero = np.argmax(matrix[i, :] != 0)
        if np.any(matrix[i + 1:, first_non_zero] != 0):
            return False
    
    return True

# Input matrix from user
R = int(input("Enter the number of rows: "))
C = int(input("Enter the number of columns: "))

matrix = []
for r in range(R):
    row = list(map(float, input(f"Enter row {r + 1} as space-separated values: ").split()))
    matrix.append(row)

matrix = np.array(matrix)

if is_rref(matrix):
    print("The matrix is in Reduced Row Echelon Form (RREF).")
else:
    print("The matrix is NOT in Reduced Row Echelon Form (RREF).")
