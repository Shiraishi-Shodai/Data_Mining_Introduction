import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

df = pd.read_csv("groupA.csv")
group_A =df[df["label_or_ypred"]==1.20].to_numpy()
group_B =df[df["label_or_ypred"]==2.20].to_numpy()

#print(group_A)
#print(group_B)
X = df[["x1","x2"]].to_numpy()

model = DBSCAN(eps=0.38, min_samples=3, metric='euclidean')
#model = KMeans(n_clusters=2, random_state=42)
model.fit(X)

cmap = plt.get_cmap("tab10")
#plt.scatter(group_A[:,0],group_A[:,1],color="red")
#plt.scatter(group_B[:,0],group_B[:,1],color="blue")
plt.scatter(X[:,0],X[:,1],c=cmap(model.labels_))
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("circle")

plt.show()
print(set(model.labels_))
