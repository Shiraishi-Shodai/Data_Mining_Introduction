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

tsne = TSNE(n_components=2, random_state=42)
tsne_reduced_data = tsne.fit_transform(digits_data)

plt.scatter(tsne_reduced_data[:,0],tsne_reduced_data[:,1],c=digits_label)
plt.savefig("tsne_0.png")
plt.show()

# dataに対してそれぞれ0か0以外の数字かを True,Flaseで返す
def is_zero(th, data, labels):
    zero_flag = True
    res=[]
    for i in range(0,len(data)):
        if data[i,1]>th:
            zero_flag = True
        else:
            zero_flag = False   
        res.append(zero_flag)
    
    return np.array(res)

#print(is_zero(40,tsne_reduced_data,digits_label))

# 以下は竹本君のコード

# 閾値の設定
threshold = 40.0

# 判別結果の取得
predictions = is_zero(threshold, tsne_reduced_data, digits_label)
print(predictions)
print("----------------------")
print(digits_label == 0)

# 正解率の計算
accuracy = accuracy_score(digits_label == 0, predictions)
print(f"正解率: {accuracy * 100:.2f}%")
