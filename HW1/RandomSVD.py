import numpy as np
import matplotlib.pyplot as plt

randomMat = np.random.randn(50,50)

U, S, VT = np.linalg.svd(randomMat, full_matrices=False)
    
plt.plot(S)
plt.xlabel("Index")
plt.ylabel("Singular Value")
plt.show()

means = []
medians = []
singularVals = []
U, S, VT = np.linalg.svd(np.random.randn(100,50,50))   



plt.boxplot(S, showfliers=False)
plt.show()

#this will show 100 tests at each rank (1 to 50) and the distrubution of the singular value at each rank in the box and whisker plot
i = 0
j = 0


for i in range (0, 50):
    means.append(np.mean(S[:,i]))
    medians.append(np.median(S[:,i]))
        
        
fig, ax1 = plt.subplots()

line1, = ax1.plot(range(1,50+1), means, color = "blue")
ax1.set_xlabel('Rank',color='black')

ax2 = ax1.twinx()
line2, = ax2.plot(range(1,50+1), medians, color = "orange")


ax1.legend([line1, line2], ["Mean", "Median"], loc="upper right")
plt.show()