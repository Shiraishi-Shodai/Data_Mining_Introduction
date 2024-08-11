import numpy as np
import matplotlib.pyplot as plt
from sklearn.semi_supervised import LabelPropagation
from sklearn.semi_supervised import LabelSpreading
import pandas as pd
import math

df = pd.read_csv('semi_super_ans.tsv', sep="\t")
X = df[["alpha", "beta"]].to_numpy()
labels = df["label"].to_numpy()

del_per = 0.8
del_num = math.floor(len(labels) * del_per)
labels[0:del_num] = np.array([-1] * del_num)

# labels_color = ["darkorange" if i < 0 else "c" if i==1 else "navy" for i in labels]
# plt.scatter(X[:, 0], X[:, 1], c=labels_color)
# plt.show()

model = LabelSpreading(kernel="knn", alpha=0.5, n_neighbors=7, max_iter=1000, n_jobs=1)
model.fit(X, labels)

labels_color2 = ["darkorange" if i < 0 else "c" if i==1 else "navy" for i in model.transduction_]
plt.scatter(X[:, 0], X[:, 1], c=labels_color2)
plt.show()
