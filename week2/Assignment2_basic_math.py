import numpy as np

# Coefficients matrix (fruit quantities)
A = np.array([
    [1, 1, 1],  # Day 1: 1 apple, 1 banana, 1 cherry
    [1, 2, 1],  # Day 2: 1 apple, 2 bananas, 1 cherry
    [1, 1, 2]   # Day 3: 1 apple, 1 banana, 2 cherries
])

# Check if the matrix is singular (determinant is zero)
det_A = np.linalg.det(A)
if det_A == 0:
    print("The matrix A is singular and not invertible.")
else:
    print(f"The determinant of A is {det_A:.2f}, so it is invertible.")
    
    # Calculate the inverse of the matrix
    A_inv = np.linalg.inv(A)
    print("The inverse of the matrix A is:")
    print(A_inv)

# Result matrix (total cost)
B = np.array([10, 15, 12])  # Day 1: $10, Day 2: $15, Day 3: $12

# Solve the system of linear equations
prices = np.linalg.solve(A, B)

# Print the results
print(f"Price of an apple: ${prices[0]}")
print(f"Price of a banana: ${prices[1]}")
print(f"Price of a cherry: ${prices[2]}")
