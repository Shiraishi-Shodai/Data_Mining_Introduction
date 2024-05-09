from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('iris.csv',usecols=['sepal_length','sepal_width','petal_length','petal_width'])
X = df.values
print(X)

distortions = []
for i  in range(1,11):
    model = KMeans(n_clusters=i,
               n_init=10,
               max_iter=300,
               random_state=0)
    model.fit(X)
    print(model.inertia_)
    distortions.append(model.inertia_) # model.inertia_は最も近いクラスター中心までのサンプルの二乗距離の合計で、提供されている場合はサンプルの重みで加重。

plt.plot(range(1,11),distortions,marker="o")
plt.xticks(range(1,11))
plt.xlabel("Number of clusters")
plt.ylabel("SSE")
plt.show()