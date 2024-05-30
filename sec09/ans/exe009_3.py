"""
* exe009_2を改良し、それぞれの軸の閾値を求める処理を追加
* TSNEは4以上の次元には圧縮できない
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

digits = datasets.load_digits()
digits_data = digits.data
#手書き文字のラベル
digits_label = digits.target
#print(digits_label)

# ラベルを使って閾値を見つける関数
def find_th(data,label,n_comp):
    th_list =[]
    for j in range(0,n_comp):
        zero_min = 10000000     # ラベルが0の内、最も低い位置
        nonzero_min = 10000000  # ラベルが0以外の内、最も低い位置
        zero_max = 0            # ラベルが0の内、最も高い位置
        nonzero_max = 0         # ラベルが0以外の内、最も高い位置
        th = 0
        for i in range(0,len(data)):
            if label[i]==0:
                if data[i,j] > zero_max:
                    zero_max = data[i,j]
                if data[i,j] < zero_min:
                    zero_min = data[i,j]
            else:#0以外の数字    
                if data[i,j] > nonzero_max:
                    nonzero_max = data[i,j]
                if data[i,j] < nonzero_min:
                    nonzero_min = data[i,j]

        if nonzero_max < zero_max:#パターンA
            th = (nonzero_max+zero_min)/2
        else:#パターンB
            th = (nonzero_min+zero_max)/2
        th_list.append(th)
    
    return th_list


# dataに対してそれぞれ0か0以外の数字かを True,Flaseで返す
def is_zero(th, data,label,n_comp):

    tsne = TSNE(n_components=n_comp, random_state=42)
    tsne_reduced_data = tsne.fit_transform(digits_data)
    print(tsne_reduced_data.shape)
    plt.scatter(tsne_reduced_data[:,0],tsne_reduced_data[:,1],c=digits_label)
    plt.savefig("tsne_3.png")
    plt.show()
    # 閾値が0より大きいときはその閾値を使用。そうでないときは、機械が閾値を探す
    if th <0:
        th_l = find_th(tsne_reduced_data,label,n_comp)
        print(th)
    res_l=[]
    for th in th_l:
        print(f"th ={th}")
        zero_flag = True
        res=[]
        for i in range(0,len(tsne_reduced_data)):
            if tsne_reduced_data[i,1]>th:
                zero_flag = True
            else:
                zero_flag = False   
            res.append(zero_flag)
        
        res_l.append(res)
    return np.array(res_l)

#print(is_zero(40,digits_data))

# 以下は竹本君のコード
# 閾値の設定
#threshold = find_th(digits_data,digits_label)
threshold =-1

# 判別結果の取得
predictions = is_zero(threshold, digits_data,digits_label,2)
print(predictions.shape)
# 正解率の計算
for pr in predictions:
    accuracy = accuracy_score(digits_label == 0, pr)
    print(f"正解率: {accuracy * 100:.2f}%")
