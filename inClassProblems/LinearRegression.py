
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['figure.figsize'] = [6, 6]
plt.rcParams.update({'font.size': 12})

x = 3 # True slope
a = np.arange(-2,2,0.25)
a = a.reshape(-1, 1)
b = x*a + np.random.randn(*a.shape) # Add noise


U, S, VT = np.linalg.svd(a,full_matrices=False)

xtilde = VT.T @ np.linalg.inv(np.diag(S)) @ U.T @ b # Least-square fit


plt.plot(a, b         , 'o', color='r',  markersize = 8, label='Noisy data') # Noisy measurements
plt.plot(a, xtilde * a,'--', color='b',     linewidth=4, label='Regression line')
plt.plot(a, x * a     ,      color='k',     linewidth=2, label='True line') # True relationship

plt.xlabel('a')
plt.ylabel('b')

plt.grid(linestyle='--')
plt.legend()
plt.show()