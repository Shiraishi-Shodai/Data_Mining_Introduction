from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples, accuracy_score
from matplotlib import cm
import numpy as np

df = pd.read_csv('003.csv')
X_train = df[["x1", "x2"]].to_numpy()
y_train = df["label"].to_numpy()
# 元データの可視化
# plt.scatter(X_train[:, 0], X_train[:, 1])
# plt.savefig('original.png')

model = KMeans(n_clusters=2, random_state=0)
model.fit(X_train)
y_train_pred = model.labels_
# print(model.labels_)
# plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train_pred)
# plt.savefig('train_data_predict.png')

"""（２）シルエット分析やグラフのプロットなどを使い、あなたの行ったクラスタリング結果が良いか悪いかを理由をつけて述べなさい。"""
"""
シルエット分析からわかること
* クラスター1もクラスター2もほとんどのデータが平均よりもシルエット係数が高くなっているので、なんとなく良く分離できてそう
* クラスター1とクラスター2のデータの厚みがだいたい同じくらいなので、同じデータ量で分離できてそう
* クラスター1はシルエット係数がマイナスになっているデータが少しあるので、そのあたりはクラスターの分離が曖昧で、密度が低くなっている

散布図からわかること
* 人間の直感とは別のところで境界線が引かれている

まとめ
少しだけ曖昧なデータが含まれているものの、ほとんどのデータは適切にクラスターを分離できてそうなので良い結果と言えそう
"""

# y_clstr = y_train_pred
# cluster_labels = np.unique(y_clstr)
# n_clusters=cluster_labels.shape[0]

# silhouette_vals = silhouette_samples(X_train,y_clstr,metric='euclidean')  # シルエット係数を計算
# y_ax_lower, y_ax_upper= 0,0
# yticks = []
# bar_color=['#CC4959','#33cc33','#4433cc']
# for i,c in enumerate(cluster_labels):
#         c_silhouette_vals = silhouette_vals[y_clstr==c]
#         c_silhouette_vals.sort()
#         y_ax_upper += len(c_silhouette_vals)
#         color = cm.jet(float(i)/n_clusters)       # 色の値を作る
#         plt.barh(range(y_ax_lower,y_ax_upper),    # 水平の棒グラフのを描画（底辺の範囲を指定）
#                          c_silhouette_vals,       # 棒の幅（1サンプルを表す）
#                          height=1.0,              # 棒の高さ
#                          edgecolor='none',        # 棒の端の色
#                          #color=color)
#                          color=bar_color[i])         # 棒の色
#         yticks.append((y_ax_lower+y_ax_upper)/2)     # クラスタラベルの表示位置を追加
#         y_ax_lower += len(c_silhouette_vals)         # 底辺の値に棒の幅を追加

# silhouette_avg = np.mean(silhouette_vals)               # シルエット係数の平均値

# plt.axvline(silhouette_avg,color="red",linestyle="--")  # 係数の平均値に破線を引く
# plt.yticks(yticks,cluster_labels + 1)                   # クラスタレベルを表示
# plt.ylabel('Cluster')
# plt.xlabel('silhouette coefficient')
# plt.savefig('silhouette.png')

"""（１）このデータを使って、次の住人を２つの体質にグループ分けしなさい。"""
# plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train_pred)
# plt.scatter(2.20, 2.89, c="red")
# plt.text(2.20, 2.89, "A")
# plt.scatter(2.02, 1.69, c="red")
# plt.text(2.02, 1.69, "B")
# plt.scatter(2.83, 1.52, c="red")
# plt.text(2.83, 1.52, "C")
# plt.scatter(1.10, 2.51, c="red")
# plt.text(1.10, 2.51, "D")
# plt.savefig('ABCD_plot.png')

X_test = np.array([
    [2.20, 2.89],
    [2.02, 1.69],
    [2.83, 1.52],
    [1.10, 2.51]
])

y_test_pred = model.fit_predict(X_test)
print(f"ABCDのクラスタリング{y_test_pred}")

"""（３）（２）まで解けた人へ。
003.csvのlabelというデータは別の検査でわかっている遺伝子型です。このラベルデータはx1,x2によるAIのクラスタリング結果とどれくらい一致しているでしょうか？一致率を出してみてください。
"""
y_pred_label = list(map(lambda x : 3 if x == 1 else 2, y_train_pred))
print(f'003.csvの一致率: {accuracy_score(y_train, y_pred_label)*100} %')


"""
(4) 実は、このデータは人工的に moonで作ったデータです。クラスタリング結果が2つのmoonをきれいに分離できていない理由を考えてみましょう。
わかりませんでした。
"""
