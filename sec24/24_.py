import numpy as np
import matplotlib.pyplot as plt

array_A =np.array([[0,0,0],[1,1,0.8]])
array_B =np.array([[0,0,0],[0.45,0.4,0.6]])

array_AA =np.array([[0,0,0],[0.5,0.4,0.6]])
array_BB =np.array([[0,0,0],[-1.0,-0.79,-1.3]])

array_AAA =np.array([[0,0,0],[-1.5,1.4,3.6]])
array_BBB =np.array([[0,0,0],[-4.5,2.9,0.6]])


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(0,0,0,color="navy")
ax.plot(array_A[:,0], array_A[:,1], array_A[:,2], color='green')
ax.plot(array_B[:,0], array_B[:,1], array_B[:,2], color='red')

ax.plot(array_AA[:,0], array_AA[:,1], array_AA[:,2], color='blue')
ax.plot(array_BB[:,0], array_BB[:,1], array_BB[:,2], color='purple')

ax.plot(array_AAA[:,0], array_AAA[:,1], array_AAA[:,2], color='yellow')
ax.plot(array_BBB[:,0], array_BBB[:,1], array_BBB[:,2], color='orange')

def cos_sim(vec1,vec2):
    return vec1@vec2/((np.linalg.norm(vec1, ord=2))*(np.linalg.norm(vec2, ord=2)))

print(cos_sim(array_A[1,:],array_B[1,:]))
print(cos_sim(array_AA[1,:],array_BB[1,:]))
print(cos_sim(array_AAA[1,:],array_BBB[1,:]))
plt.show()
