import numpy as np
import matplotlib.pyplot as plt

randomMat = np.random.rand(50,50)

U, S, VT = np.linalg.svd(randomMat, full_matrices=False)

print(S)
    
plt.plot(S, marker='o')
plt.xlabel("Index")
plt.ylabel("Singular Value")
plt.show()