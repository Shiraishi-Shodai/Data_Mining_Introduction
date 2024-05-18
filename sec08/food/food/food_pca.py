from food import *
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.decomposition import PCA

png_data ,label = dir2tensor("./jpg")
#print(png_data)
print(label)
label = -1*label +2 
model = PCA(n_components=2)
model.fit(png_data)
X_pca = model.transform(png_data)
plt.scatter(X_pca[:,0],X_pca[:,1],c=label)
plt.title("クラスタリングによるデータのグループ化")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

from sklearn.cluster import KMeans
kmeans_model = KMeans(n_clusters=2, random_state=10).fit(X_pca)
labels = kmeans_model.labels_
#print(labels)
plt.scatter(X_pca[:,0],X_pca[:,1],c=labels)
plt.title("本当の寿司とフライドポテト")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()
