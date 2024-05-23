import numpy as np
import matplotlib.pyplot as plt
from sklearn.semi_supervised import LabelSpreading
import pandas as pd
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.svm import SVC
#df = pd.read_csv("m_q.txt") #moon

#df = pd.read_csv("semi_super_ans.tsv",sep="\t") #二重の円
df = pd.read_csv("semi_super_noisey.tsv",sep="\t") #二重の円

X = df[['alpha','beta']].values
labels = df['label'].values
labels[0:350] = np.array([-1]*350) #388行のうち350行をunknownにする。
svc = SVC(probability=True, gamma="auto")
self_training_model = SelfTrainingClassifier(svc)
self_training_model.fit(X[351:], labels[351:])
predict_y = self_training_model.predict(X[0:350])

print(predict_y)
labels_color = [('darkorange' if i < 0 else ('c' if i == 1 else 'navy')) for i in predict_y]
plt.scatter(X[0:350, 0], X[0:350, 1], c=labels_color)
plt.title("after fit with semi_supervised(SL)")
plt.xlabel("alpha")
plt.ylabel("beta")
plt.savefig("SL.png")
plt.show()


