import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.cluster import DBSCAN

df = pd.read_csv('group_sample.csv')
X = df[['x1','x2']].to_numpy()
y_label = df['y'].to_numpy()

model = DBSCAN(eps=0.3, min_samples=19, metric='euclidean')
model.fit(X)

#model.labels_からノイズを除いてクラスター数は何個あるか。
def get_cluster_num(label):
    #ノイズがあるとき
    if -1 in label:
        return len(list(set(label)))-1
    #ノイズがないとき
    else:
        return len(list(set(label)))

#2つのリストの対応する要素が一致するものの割合を求める
#対応するラベルが違うと一致率が正しく出ない
def list_accuracy(list1,list2):
    m =0
    for i in range(0,len(list1)):
        if list1[i]==list2[i]:
            m +=1
    
    return m/len(list1)

print(get_cluster_num(model.labels_))

plt.scatter(X[:,0], X[:,1],c=y_label)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('人間のラベルによる色分け')
plt.savefig('exe007_label.png')
plt.show()

plt.scatter(X[:,0], X[:,1],c=model.labels_)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('人間のラベルによる色分け')
plt.savefig('exe007_demo.png')
plt.show()
