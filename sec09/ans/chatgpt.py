import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
"""
機械学習でよく使われるdigitsというデータがあります。
これは手書きの数字のデータセットで、scikit-learnではdatasets.load_digits()で呼び出して使うことができます。このデータの数字の0かそれ以外の数字かということを判別する関数をpythonで書いてください。ただし、ラベルデータは使わずに次元削減やクラスタリングを用いてください。
"""
# データセットを読み込み
digits = load_digits()
X = digits.data

# 次元削減（PCAを使用）
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# K-meansクラスタリング
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X_pca)

# クラスタリング結果の可視化
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis')
plt.colorbar()
plt.title('Clustering of Digits Data (0 vs. Others)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()

# クラスタごとに0の割合を計算
is_zero = (digits.target == 0)
cluster_labels = kmeans.labels_
zero_in_cluster_0 = is_zero[cluster_labels == 0].mean()
zero_in_cluster_1 = is_zero[cluster_labels == 1].mean()

print(f"Cluster 0 has {zero_in_cluster_0 * 100:.2f}% zeros")
print(f"Cluster 1 has {zero_in_cluster_1 * 100:.2f}% zeros")

# 関数化
def classify_zero(X):
    X_pca = pca.transform(X)
    clusters = kmeans.predict(X_pca)
    return clusters

# 数字が0である確率が高いクラスタの特定
if zero_in_cluster_0 > zero_in_cluster_1:
    zero_cluster = 0
else:
    zero_cluster = 1

# 0かそれ以外かを判別する関数
def is_zero_digit(digit):
    cluster = classify_zero(digit.reshape(1, -1))
    return cluster == zero_cluster

# テスト
for i in range(10):
    digit = X[i]
    print(f"Digit {digits.target[i]} classified as {'0' if is_zero_digit(digit) else 'non-0'}")
