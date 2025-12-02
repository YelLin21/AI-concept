# import numpy as np

# # Define matrix A
# A = np.array([[3, 0, 2],
#               [2, 0, -2],
#               [0, 1, 1]])

# # Calculate the determinant of A
# det = (A[0,0]*(A[1,1]*A[2,2] - A[1,2]*A[2,1]) -
#        A[0,1]*(A[1,0]*A[2,2] - A[1,2]*A[2,0]) +
#        A[0,2]*(A[1,0]*A[2,1] - A[1,1]*A[2,0]))

# # Check if the matrix is invertible
# if det == 0:
#     print("Matrix A is singular and does not have an inverse.")
# else:
#     # Calculate the adjugate (cofactor matrix)
#     adjugate = np.array([
#         [A[1,1]*A[2,2] - A[1,2]*A[2,1], -(A[0,1]*A[2,2] - A[0,2]*A[2,1]), A[0,1]*A[1,2] - A[0,2]*A[1,1]],
#         [-(A[1,0]*A[2,2] - A[1,2]*A[2,0]), A[0,0]*A[2,2] - A[0,2]*A[2,0], -(A[0,0]*A[1,2] - A[0,2]*A[1,0])],
#         [A[1,0]*A[2,1] - A[1,1]*A[2,0], -(A[0,0]*A[2,1] - A[0,1]*A[2,0]), A[0,0]*A[1,1] - A[0,1]*A[1,0]]
#     ])

#     # Calculate the inverse by dividing the adjugate by the determinant
#     A_inv = adjugate / det
#     print("Inverse of A:\n", A_inv)

import numpy as np

A = np.array([[3, 0, 2], [2, 0, -2],[0, 1, 1]])

A_inv = np.linalg.inv(A)
print("Inverse of A:\n", A_inv)

#Multiply A with A_inv to get the identity matrix
I = np.dot(A, A_inv)
print("Identity Matrix:\n", I)

I = np.round(I, decimals=6)

print("Identity Matrix:\n", I)