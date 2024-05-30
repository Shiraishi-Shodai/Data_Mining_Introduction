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

# データの読み込み
digits = datasets.load_digits()
digits_data = digits.data
digits_label = digits.target

# ラベルを使って閾値を見つける関数
def find_th(data,label):
    zero_min = 10000000
    nonzero_min = 10000000
    zero_max = 0
    nonzero_max = 0
    for i in range(0,len(data)):
        if label[i]==0:
            if data[i,1] > zero_max:
                zero_max = data[i,1]
            if data[i,1] < zero_min:
                zero_min = data[i,1]
        else:#0以外の数字    
            if data[i,1] > nonzero_max:
                nonzero_max = data[i,1]
            if data[i,1] < nonzero_min:
                nonzero_min = data[i,1]

    if nonzero_max < zero_max:#パターンA
        th = (nonzero_max+zero_min)/2
    else:#パターンB
        th = (nonzero_min+zero_max)/2
    return th


# dataに対してそれぞれ0か0以外の数字かを True,Flaseで返す
def is_zero(th, data,label):

    tsne = TSNE(n_components=2, random_state=42)
    tsne_reduced_data = tsne.fit_transform(digits_data)

    plt.scatter(tsne_reduced_data[:,0],tsne_reduced_data[:,1],c=digits_label)
    plt.savefig("tsne_1.png")
    plt.show()
    if th <0:
        th = find_th(tsne_reduced_data,label)
    print(f"th ={th}")
    zero_flag = True
    res=[]
    for i in range(0,len(tsne_reduced_data)):
        if tsne_reduced_data[i,1]>th:
            zero_flag = True
        else:
            zero_flag = False   
        res.append(zero_flag)
    
    return np.array(res)

#print(is_zero(40,digits_data))

# 以下は竹本君のコード
# 閾値の設定
#threshold = find_th(digits_data,digits_label)
threshold =-1

# 判別結果の取得
predictions = is_zero(threshold, digits_data,digits_label)
print(predictions)
print("----------------------")
print(digits_label == 0)

# 正解率の計算
accuracy = accuracy_score(digits_label == 0, predictions)
print(f"正解率: {accuracy * 100:.2f}%")
