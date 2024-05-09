from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib

df = pd.read_csv('group_sample.csv')
X = df[['x1','x2']].to_numpy()
y_label = df['y'].to_numpy()

distortions = []
for i in range(1, 11):
    model = KMeans(n_clusters=i,
                   n_init=10,
                   max_iter=300,
                   random_state=0)
            
    model.fit(X)
    distortions.append(model.inertia_)


plt.plot(range(1, 11), distortions, marker='o')
plt.xticks(range(1,11))
plt.xlabel("Number of clusters")
plt.ylabel("SSE")
model = KMeans(n_clusters=3, random_state=0)
model.fit(X)
print(model.labels_)
plt.scatter(X[:,0],X[:,1],c=model.labels_)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title("サンプルデータ")
plt.show()

