"""
class LabelSpreading(
    kernel: ((...) -> Any) | Literal['rbf', 'knn'] = "rbf", rbfはガウシアン knnはk近傍法
    *,
    gamma: Float = 20,
    n_neighbors: Int = 7, knnを使う時に使う
    alpha: Float = 0.2,
    max_iter: Int = 30,
    tol: Float = 0.001,   システムが定常状態になったと判断するしきい値
    n_jobs: Int | None = None 実行する並列ジョブの数
)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.semi_supervised import LabelPropagation
from sklearn.semi_supervised import LabelSpreading
import pandas as pd

#df = pd.read_csv("semi_moon.tsv",sep"\t") #moon
df = pd.read_csv("semi_super_noisey.tsv",sep="\t") #二重の円

X = df[['alpha','beta']].values
labels = df['label'].values
t_n= 367
labels[0:t_n] = np.array([-1]*t_n) #388行のうち350行をunknownにする。
#print(labels)
labels_color = [('darkorange' if i < 0 else ('c' if i == 1 else 'navy')) for i in labels]
plt.scatter(X[:, 0], X[:, 1], c=labels_color)
plt.title("bofore fit with semi_supervised")
plt.xlabel("alpha")
plt.ylabel("beta")
plt.show()

model = LabelSpreading(kernel='knn', alpha=0.2, n_neighbors=7,max_iter=1000, n_jobs=-1)
model.fit(X, labels)
#print(model.transduction_)

labels_color2 = [('darkorange' if i < 0 else ('c' if i == 1 else 'navy')) for i in model.transduction_]
plt.scatter(X[:, 0], X[:, 1], c=labels_color2)
plt.title("after fit with semi_supervised")
plt.xlabel("alpha")
plt.ylabel("beta")
plt.show()