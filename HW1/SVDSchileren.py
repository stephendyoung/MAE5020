import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from PIL import Image
from urllib.request import urlopen



url = "https://upload.wikimedia.org/wikipedia/commons/9/94/F4_p3_rgb_planedrop.jpg"

image = Image.open(urlopen(url))

image = image.convert('L')

X = np.array(image)

plt.imshow(X, cmap='gray')

plt.show()

print("Dimension of X :", X.ndim)
print("Shape of X     :", X.shape)
print("Size of X      :", X.size)
print("Min(X)         :", np.min(X))
print("Max(X)         :", np.max(X))
print("X.             :\n",X)

U, S, VT = np.linalg.svd(X, full_matrices=False)
print("Shape of U     :", U.shape)
print("Shape of S     :", S.shape)
print("Shape of VT    :", VT.shape)

S = np.diag(S)
print("np.diag(S)     :", S.shape)





plt.figure(1)
plt.plot(np.cumsum(np.diag(S))/np.sum(np.diag(S)))
plt.axvline(443, color='red', linestyle='dashed')
plt.axhline(0.90,color='black', linestyle='dashed')
plt.title('Singular Values: Cumulative Sum')
plt.xlim(0,1000)
plt.show()








# j = 0
# for r in (10, 20, 40):
#     Xapprox =  U[:,:r] @ S[0:r,:r] @ VT[:r,:]
#     print("%%%%%%%%%%%%%%%%%%%%%")
#     print("--------------r:",r)
#     print("Shape of U     :", U[:,:r].shape)
#     print("Shape of S     :", S[0:r,:r].shape)
#     print("Shape of VT    :", VT[:r,:].shape)
#     print("Shape of Xaprx :", Xapprox.shape)
    
    
        
    
#     plt.figure(j+1)
#     j +=1
#     img = plt.imshow(Xapprox)
#     img.set_cmap('gray')
#     plt.axis('off')
#     plt.title('r = ' + str(r))
#     plt.show()       

r = 17
Ur = U[:, :r]
UTUTransposeCheck = Ur @ np.transpose(Ur)
print(f"U*U^T is rxr check: {UTUTransposeCheck.shape}")


r = 17
Ur = U[:, :r]
UTUTransposeCheck =np.transpose(Ur) @ Ur 
print(f"U^T*U is not rxr check: {UTUTransposeCheck.shape}")


n = X.shape[1]
i = 0
j=0

error = np.zeros((n,n))
for i in range(1, n+1):
    identityMat = np.eye(i)
    # Ur = U[:,:i]
    # UUT = np.transpose(Ur) @ Ur
    
print(n)
print(error)
plt.plot(error.T)
plt.xlabel("rank")
plt.ylabel("error")




