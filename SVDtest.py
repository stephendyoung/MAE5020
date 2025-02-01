import numpy as np

# 2x3 matrix
A = np.array([[3, 2, 2],
              [2, 3, -2]])

# Apply SVD
U, S, V = np.linalg.svd(A)

print("Matrix A:")
print(A)
print("\nU:")
print(U)
print("\nS:")
print(S)
print("\nV:")
print(V)


