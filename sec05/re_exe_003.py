import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import japanize_matplotlib
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from matplotlib import cm
from sklearn.metrics import accuracy_score

df = pd.read_csv('003.csv')
X = df[["x1", "x2"]].to_numpy()
y = df["label"].to_numpy()

data = np.array([
[2.20,2.89],
[2.02,1.69],
[2.83,1.52],
[1.10,2.51]
])

model = KMeans(n_clusters=2, random_state=42)
model.fit(X)

# plt.scatter(data[0, 0], data[0, 1], c="red")
# plt.text(data[0, 0], data[0, 1], "Aさん")

# plt.scatter(data[1, 0], data[1, 1], c="red")
# plt.text(data[1, 0], data[1, 1], "Bさん")

# plt.scatter(data[2, 0], data[2, 1], c="red")
# plt.text(data[2, 0], data[2, 1], "Cさん")

# plt.scatter(data[3, 0], data[3, 1], c="red")
# plt.text(data[3, 0], data[3, 1], "Dさん")
# plt.scatter(X[:, 0], X[:, 1], c=model.labels_)
# plt.savefig('original.png')

y_clstr = model.labels_

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
plt.savefig('silhouette.png')

"""
クラスター2はシルエット係数がいくつか負の値になっているので、これらのデータは密度が低く間違って分類されてしまっていそう
クラスター1も2も半分くらのデータが平均より大きなシルエット係数をとっている
クラスター1と2の厚みは同じくらいなのでだいたい均等にクラスタリングされていそう
誤分類が少々あると考えられるものの、だいたい良いクラスタリング結果だと考えられる
"""

"""
(3)　(2)まで解けた人へ。
003.csv の label というデータは別の検査でわかっている遺伝子型です。このラベルデータは x1, x2 による AI のクラスタリング結果とどれくらい一致しているでしょうか？一致率を出してみてください。
"""
y_pred = list(map(lambda x: 3 if x == 0 else 2, model.labels_))
print(y)
print(y_pred)
print(f"正解率: {accuracy_score(y, y_pred)}")