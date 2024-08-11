import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv('iris.csv')
X = df.iloc[:, :-1].to_numpy()
y = df["species"].to_numpy()

distortions = []
for i in range(1, 11):
    model = KMeans(n_clusters=i, n_init=10, max_iter=300, random_state=0)
    model.fit(X)
    distortions.append(model.inertia_)

plt.plot(range(1, 11), distortions, marker='o')
plt.xticks(range(1, 11))
plt.xlabel("Number of clusters")
plt.ylabel('SSE')
plt.show()
