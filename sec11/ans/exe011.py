import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from sklearn.semi_supervised import LabelSpreading
from sklearn.svm import SVC

df = pd.read_csv("iris.csv",sep=",") 
df = df.sample(frac=1)
X = df[['sepal_length','sepal_width','petal_length','petal_width']].to_numpy()
labels = df['species'].map({'setosa':0, 'versicolor':1,'virginica':2}).to_numpy()
#print(X)
#print(labels)
labels_org = labels.copy() #ラベルを消す前のシャッフルだけしたラベル

rate = 0.08 # 0.05 0.08
t_n= int(len(X)*(1-rate))
labels[0:t_n] = np.array([-1]*t_n) #rateの分だけを残したラベルがほとんどないラベル
print(labels)

#教師あり学習と半教師あり学習の結果の比較
def comp_model(X,labels,t_n):
    model1 = LabelSpreading(kernel='knn', alpha=0.8, n_neighbors=7,max_iter=1000, n_jobs=-1)
    model1.fit(X, labels)
    result1 = model1.predict(X)

    model2 = SVC(gamma="scale")
    model2.fit(X[t_n:,:], labels[t_n:])
    result2 = model2.predict(X)
    return [result1,result2]

#2つの配列の一致率を調べる
def accuracy_array(array_a,array_b):
    array_c = (array_a==array_b)
    return int(np.count_nonzero(array_c)/len(array_c)*100)

# メイン
res1,res2 = comp_model(X,labels,t_n)

print("\n---オリジナルのシャッフル後ラベル-------------")
print(labels_org)
print("\n---ラベル削除後-------------")
print(labels)
print("\n---半教師あり学習（ラベル拡散）-------------")
print(res1)
print("\n---SVM（SVC）-------------")
print(res2)

print(f"ラベル拡散による結果:",end="")
print(f"{accuracy_array(res1,labels_org)}",end="%\n")
print(f"SVMによる結果:",end="")
print(f"{accuracy_array(res2,labels_org)}",end="%\n")
