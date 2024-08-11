from sklearn import random_projection
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import japanize_matplotlib

digits = datasets.load_digits()
# print(digits.data[0].)
plt.imshow(digits.data[0].reshape(8, 8), cmap='gray')
plt.show()
X_TSNEprojected = TSNE(n_components=2, random_state=0).fit_transform(digits.data)

plt.scatter(X_TSNEprojected[:,0], X_TSNEprojected[:,1], c=digits.target,alpha=0.5, cmap='rainbow')
plt.colorbar()
# plt.show()
plt.savefig('08_25_01.png')

print(digits.data.shape)
print(X_TSNEprojected.shape)