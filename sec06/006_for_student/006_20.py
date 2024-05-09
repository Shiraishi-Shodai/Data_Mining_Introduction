from sklearn.cluster import DBSCAN
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('iris.csv',usecols=['sepal_length','sepal_width','petal_length','petal_width'])
X = df.to_numpy()
print(X)

model = DBSCAN(eps=0.4, min_samples=5, metric='euclidean')

model.fit(X)

for i in range(0, len(model.labels_)):
    print(model.labels_[i])

plt.scatter(df['sepal_width'],df['petal_length'],c=model.labels_)
plt.xlabel('sepal_width')
plt.ylabel('petal_length')
# plt.savefig('006_20.png')
plt.show()
print(set(model.labels_))