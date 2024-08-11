from sklearn.cluster import DBSCAN
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('iris.csv', usecols=['sepal_length','sepal_width','petal_length','petal_width'])
X = df.to_numpy()

# ①　eps		:一方が他方の近傍にあるとみなされる 2 つのサンプル間の最大距離。
# ②　min_samples	:ポイントがクラスタと見なされる近傍内のサンプル数
model = DBSCAN(eps=0.4, min_samples=5, metric="euclidean")

model.fit(X)

cmap = plt.get_cmap("Paired")
plt.scatter(X[:, 1], X[:, 2], c=cmap(model.labels_))
plt.show()
print(set(model.labels_))