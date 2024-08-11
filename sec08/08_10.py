import matplotlib.pyplot as plt
import random
import numpy as np
import math
import japanize_matplotlib

"""データを回転するコード"""

def rotate_point(x,theta):
    R_matrix =np.array([
        [math.cos(theta) , -math.sin(theta)],
        [math.sin(theta) , math.cos(theta) ]
    ])
    vec_X = R_matrix@x
    return vec_X

x = 20* np.random.random(200)
y = np.random.uniform(-1,1,200)
x = x.reshape(-1,1)
y = y.reshape(-1,1)*0.4
X = np.concatenate([x, y], 1)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(0, 21)
ax.set_ylim(-10, 10)
ax.set_xlabel("α")
ax.set_ylabel("β")
ax.scatter(X[:,0],X[:,1],color="blue")
plt.savefig('08_01.png')
# plt.show()

np.savetxt('org.csv',X,fmt='%.5e',delimiter=',')
X_new = np.array([])
for x_i in X:
    x_ii = rotate_point(x_i,3.1415926/6)
    print(x_ii)
    X_new = np.append(X_new, x_ii)
    plt.scatter(x_ii[0],x_ii[1],color="green")
plt.savefig('08_02.png')
# plt.show()

#np.savetxt('sample.csv',X_new.reshape(-1,2),fmt='%.5e',delimiter=',')
np.savetxt('sample.csv',X_new.reshape(-1,2),delimiter=',')
