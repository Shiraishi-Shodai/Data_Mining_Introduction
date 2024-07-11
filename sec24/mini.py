import numpy as np
from matplotlib import pyplot as plt
import japanize_matplotlib

# ①
a = np.array([1, 1, 0.8])
b = np.array([0.45, 0.4, 0.6])

# ②
c=np.array([0.5, 0.4, 0.6])
d=np.array([-1.0, -0.79, -1.3])

# ③
e=np.array([-1.5, 1.4, 3.6])  
f=np.array([-4.5, 2.9, 0.6])

pairs = np.array([
    [[1, 1, 0.8], [0.45, 0.4, 0.6]],
    [[0.5, 0.4, 0.6], [-1.0, -0.79, -1.3]],
    [[-1.5, 1.4, 3.6], [-4.5, 2.9, 0.6]],
])

# コサイン類似度を求める関数
def get_cos_ruijido(a, b):
    
    # norm_a =  np.sqrt(np.sum(np.abs(a**2)))
    # norm_b =  np.sqrt(np.sum(np.abs(b**2)))
    
    # または
    norm_a = np.linalg.norm(a, ord=2)
    norm_b = np.linalg.norm(b, ord=2)
    return a @ b / (norm_a * norm_b)


for v in pairs:
    a = v[0]
    b = v[1]
    print(get_cos_ruijido(a, b))


# 3次元プロット
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
x=np.zeros(2)
y=np.zeros(2)
z=np.zeros(2)
x[0], y[0], z[0] = a[0], a[1], a[2]
x[1], y[1], z[1] = b[0], b[1], b[2]
u, v, w = 0.3, 0.3, 0.3
ax.quiver(x,y,z,u,v,w,arrow_length_ratio=0.2)
ax.scatter(x, y, z, label="(x, y, z)")

ax.set_xlim(1.0)
ax.set_ylim(1.0)
ax.set_zlim(1.0)
plt.show()