import numpy as np
import matplotlib.pyplot as plt
#from sklearn.semi_supervised import LabelPropagation
#from sklearn.semi_supervised import LabelSpreading
import pandas as pd
from sklearn.svm import SVC

#df = pd.read_csv("semi_moon.tsv",sep"\t") #moon
df = pd.read_csv("semi_super_ans.tsv",sep="\t") #二重の円

X = df[['alpha','beta']].values
labels = df['label'].values
t_n= 367
labels[0:t_n] = np.array([-1]*t_n) #388行のうち350行をunknownにする。
#print(labels)
labels_color = [('darkorange' if i < 0 else ('c' if i == 1 else 'navy')) for i in labels]
plt.scatter(X[:, 0], X[:, 1], c=labels_color)
plt.title("bofore fit with SVM")
plt.xlabel("alpha")
plt.ylabel("beta")
plt.show()

model = SVC(gamma="scale")
#model = SVC(C=0.2)
model.fit(X[t_n:,:], labels[t_n:])
print("training data is ",end="")
print(X[t_n:,:])
print("training data (label )is ",end="")
print(labels[t_n:])
result = model.predict(X)

labels_color2 = [('darkorange' if i < 0 else ('c' if i == 1 else 'navy')) for i in result]
print("model's answer is  ",end="")
print(result)
plt.scatter(X[:, 0], X[:, 1], c=labels_color2)
plt.title("after fit with SVM")
plt.xlabel("alpha")
plt.ylabel("beta")
plt.show()