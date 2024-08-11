import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv('group_sample.csv')
X = df[["x1", "x2"]].to_numpy()
y = df["y"].to_numpy()

# plt.scatter(X[:, 0], X[:, 1], c="navy")
# plt.show()

model = KMeans(n_clusters=3, random_state=42)
model.fit(X)
y_pred = model.labels_

plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()
print(y)

y_pred = list(map(lambda x : 1 if x == 2 else 2 if x == 0 else 3, y_pred))