from sklearn.cluster import DBSCAN
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import numpy as np

def get_cluster_num(X,y):
    #eps は0.1～3.0の範囲で min_samplesは2～20とする。
    return_dat =[]
    for eps_i in np.arange(0.1,3.1,0.1):
        for min_sample_i in range(2,21):
            model = DBSCAN(eps=eps_i, min_samples=min_sample_i, metric='euclidean')
            model.fit(X)
            noise_flag = 1 #ノイズの有無
            cluster_num = len(list(set(model.labels_)))-1
            if -1 in set(model.labels_):
                cluster_num = len(list(set(model.labels_)))-1
            else:
                noise_flag = 0
            #print(set(model.labels_))
            c_label = model.labels_.tolist().copy()
            if cluster_num ==3:
                #print(c_label)
                #return_dat.append([eps_i, min_sample_i,cluster_num,noise_flag,c_label])
                return_dat.append([eps_i, min_sample_i,cluster_num,noise_flag])
                #return_dat.append([eps_i, min_sample_i,cluster_num,noise_flag])
                acc =list_accuracy(model.labels_,y)
                plt.title(f"クラスタリング結果 eps={eps_i:.2f} min_samples={min_sample_i:.2f} 正解率={acc}")
                plt.scatter(X[:,0],X[:,1],c=cmap2(model.labels_))
                plt.savefig(f"result{eps_i:.3f}_{min_sample_i}_.png")
                #plt.show()
                plt.close()
    #return_dat.append([0.39,5,3,1])
    return return_dat

#色分け用
def cmap2(c_list):
    c_map =["orange","red","green","blue","black"]
    colors =[]
    for i in range(0,len(c_list)):
        colors.append(c_map[c_list[i]])
    
    return colors

#2つのリストの対応する要素が一致するものの割合を求める
def list_accuracy(list1,list2):
    m =0
    for i in range(0,len(list1)):
        if list1[i]==list2[i]:
            m +=1
    
    return m/len(list1)

df = pd.read_csv('group_sample.csv')
X = df[['x1','x2']].to_numpy()
y = df["y"].to_numpy()-1
plt.title("オリジナルデータによる分類")
plt.scatter(X[:,0],X[:,1],c=cmap2(y))
plt.savefig("original.png")
plt.show()

get_cluster_num(X,y)

