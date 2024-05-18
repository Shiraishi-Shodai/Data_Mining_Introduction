# 2024年5月9日演習問題

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import japanize_matplotlib
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import DBSCAN, KMeans
from typing import List
from sklearn.metrics import accuracy_score, silhouette_samples
from skfuzzy.cluster import cmeans
from matplotlib import cm

def countCluster(labels):
    """クラスター数をカウント。外れ値があれば-1を引く
    """
    if -1 in set(labels):
        return len(list(set(labels))) -1
    else:
        return len(list(set(labels)))

def list_accuracy(list1, list2):
    m = 0
    for i in range(0, len(list1)):
        if list1[i] == list2[i]:
            m += 1
    
    return m/len(list1)    

cmap = plt.get_cmap("Set1")
df = pd.read_csv('group_sample.csv')
X = df[["x1", "x2"]].to_numpy()
y = df["y"].to_numpy()

# plt.scatter(X[:, 0], X[:, 1], c=cmap(y))
# plt.savefig('output/original.png')

"""
(1) scikit-learn の DBSCAN クラスを使い、eps,  min_samples をいろいろな数に変えて、3 クラスタにできる eps,  min_samples の値を探しなさい。( labels_ が -1 の値はノイズ（判定不明）である。多少ノイズが混じるのはしかたない。ノイズ以外で 3 クラスタ)

eps は 0.1～3.0 の範囲で min_samples は 2～20 とする。
そのときのクラスター分けした様子を色分けした散布図にせよ。

"""
print('=======================DBSCAN=======================')

best_params: dict[str, float]= {
    "eps": 0.0,
    "min_samples": 0.0
}
best_score = 0.0

for eps in np.arange(0.1, 3.1, 0.01):
    for min_samples in np.arange(2, 21, 1):
        model = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
        model.fit(X, y)
        labels = list(map(lambda label: label if label==-1 else label + 1, model.labels_)) # ラベルエンコーディング
        score = list_accuracy(y, labels)
        if countCluster(labels) == 3 and best_score < score:
            best_score = score
            best_params.update(eps=eps, min_samples=min_samples)
            print(f'現在のベストスコア{score}')
            print(f'現在のベストパラメータ {best_params}')

"""
(2) CSV のラベルデータと比較し、あなたが（1）で決定したeps,  min_samples のときの正解率をラベルのデータ（CSVの y 列）から算出せよ。もっとも高くて何％くらいまで行けるか。
"""
bestModel = DBSCAN(eps=best_params["eps"], min_samples=best_params["min_samples"], metric='euclidean')
bestModel.fit(X)
labels = list(map(lambda label: label if label==-1 else label + 1, bestModel.labels_)) # ラベルエンコーディング
# # plt.scatter(X[:, 0], X[:, 1], c=cmap(labels))
# # plt.savefig('output/DBSCAN.png')

# # print(y)
# # print()
# # print(bestModel.labels_)
# # print(set(bestModel.labels_))
# print()
print(f'最終的なベストパラメータ {best_params}')
print(f'最終的な最高正解率 : {list_accuracy(y, labels)*100} %')

"""
おまけ1: Fuzzy C-meansを使うとどうなるのか？
"""
def target_to_color(target):
    if type(target) == np.ndarray:
        return (target[0], target[1], target[2]) # rgb
    else:
        return "rgb"[target]

m = 2.0
c_means = cmeans(X.T, 3, m, 0.003, 10000)
# plt.scatter(X[:, 0], X[:, 1], c=[target_to_color(t) for t in c_means[1].T])
# plt.savefig("output/fuzzy_cmeans.png")

"""
おまけ2: KMeansを使うとどうなるのか？
"""

# エルボー法を使って妥当なクラスター数を可視化
distortions = []
for i in range(1, 11):
    model = KMeans(
        n_clusters=i,
        n_init=10,
        max_iter=300,
        random_state=0
    )
    
    model.fit(X)
    distortions.append(model.inertia_)

# plt.plot(range(1, 11), distortions, marker="o")
# plt.xticks(range(1, 11))
# plt.xlabel('Number of clusters')
# plt.ylabel('SSE')
# plt.savefig('output/elbow.png')

model = KMeans(n_clusters=3, random_state=0)
y_clstr = model.fit_predict(X)
y_clstr = list(map(lambda x: 1 if x == 2 else 2 if x == 0 else 3, y_clstr))
# print(y)
# print(y_clstr)
print()
print('=======================Kmeans=======================')
print(f'正解率: {list_accuracy(y, y_clstr)*100} %')
# plt.scatter(X[:, 0], X[:, 1], c=cmap(y_clstr))
# plt.savefig('putput/Kmeans.png')

#以下はS.Raschka他『Pythoh機械学習プログラミング　第３版』（インプレス）のコードを引用している
cluster_labels = np.unique(y_clstr)
n_clusters=cluster_labels.shape[0]

silhouette_vals = silhouette_samples(X,y_clstr,metric='euclidean')  # シルエット係数を計算
y_ax_lower, y_ax_upper= 0,0
yticks = []
bar_color=['#CC4959','#33cc33','#4433cc']
for i,c in enumerate(cluster_labels):
        c_silhouette_vals = silhouette_vals[y_clstr==c]
        c_silhouette_vals.sort()
        y_ax_upper += len(c_silhouette_vals)
        color = cm.jet(float(i)/n_clusters)       # 色の値を作る
        plt.barh(range(y_ax_lower,y_ax_upper),    # 水平の棒グラフのを描画（底辺の範囲を指定）
#つづき
                         c_silhouette_vals,       # 棒の幅（1サンプルを表す）
                         height=1.0,              # 棒の高さ
                         edgecolor='none',        # 棒の端の色
                         #color=color)
                         color=bar_color[i])         # 棒の色
        yticks.append((y_ax_lower+y_ax_upper)/2)     # クラスタラベルの表示位置を追加
        y_ax_lower += len(c_silhouette_vals)         # 底辺の値に棒の幅を追加

silhouette_avg = np.mean(silhouette_vals)               # シルエット係数の平均値

plt.axvline(silhouette_avg,color="red",linestyle="--")  # 係数の平均値に破線を引く
plt.yticks(yticks,cluster_labels + 1)                   # クラスタレベルを表示
plt.ylabel('Cluster')
plt.xlabel('silhouette coefficient')
plt.savefig('output/silhouette.png')

"""
おまけ3: KMeansの予測値のシークレット図からなにがわかる？

* 平均の赤線よりシークレット係数が大きいデータが多いので、うまくクラスタリングできてそう
* ただ、緑のデータ群は半分くらいのデータが平均よりシークレット係数が小さいのでこのクラスターは隣接しているクラスターとの距離が近いデータが多そう
* シークレット係数が負の値になっているデータはないので、ある程度各クラスターの密度が高く、ある程度明確にクラスタリング出来てそう
* 緑のクラスターが他のクラスタート比べて圧倒的にデータ量が多い
"""
