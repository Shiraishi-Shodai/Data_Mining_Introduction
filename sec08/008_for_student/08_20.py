
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

"""データを回転するコード"""

df = pd.read_csv('sample.csv',header=None,names=['x1','x2'])
X = df[['x1','x2']].to_numpy()
plt.scatter(X[:,0],X[:,1])
plt.savefig('08_20_01.png')
# plt.show()


model_pca = PCA(n_components=1)
vecs_list = model_pca.fit_transform(X)
vecs_list = vecs_list + 10.0
#print(vecs_list)
plt.scatter(vecs_list[:,0],np.zeros(len(vecs_list)))
plt.savefig('08_20_02.png')
# plt.show()
import math
print(math.sqrt(10**2+17**2))

df = pd.read_csv('org.csv',header=None,names=['x3','x4'])
X_o = df[['x3','x4']].values
print(X_o- vecs_list)
