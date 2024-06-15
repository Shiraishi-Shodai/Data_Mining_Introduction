import sympy
import numpy as np

data_ex=np.array([
    [1.0,3.1],
    [2.0,5.1],
    [3.0,6.8],
    [4.0,8.9],
    [5.0,11.5],
    [6.0,13.8],
])

#行列Xを作る
X = np.concatenate([np.ones((data_ex.shape[0], 1)),data_ex[:,0].reshape(-1,1)],1)
print(X)

t = data_ex[:,1].copy()
print(t)
w = np.linalg.inv(X.T @ X) @ X.T @ t
print(w)

print(f"x=7の時の予測値は{w[0]+w[1]*7.0}")
