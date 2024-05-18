# 2024/05/16


import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
from PIL import Image
import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score
import cv2

"""
(1)教師あり学習を使わずに、何かしらの閾値だけで判断できないだろうか。教師なし学習だけを使って digits の中から「0」とそれ以外の数字を判別する関数 is_zero を完成させなさい。(以下のコードを完成させなさい)
ヒント：次元削減
"""
# digitsデータを閾値th でdataから0かそれ以外の数字化を判定する関数
def is_zero(th,data):
    if th <= data:
        return "zero"
    else:
        return "other"

digits = datasets.load_digits()
data = digits.data
label = digits.target

"""
(2）digits データのラベルデータ（digits.target）を使い、あなたの考えた
0 の判別関数が、何％程度 0 を正しく判定できたか、その正解率を求めなさい。
"""
th, im_th_tz = cv2.threshold(data, 1, 255, cv2.THRESH_BINARY) # 二値化1より大きい数は255とするそれ以外の値は0とする
normalize_data = im_th_tz / 255.0 # 正規化(0~1)
# X_TSNEprojected = PCA(n_components=2, random_state=0).fit_transform(normalize_data)
X_TSNEprojected = umap.UMAP(n_components=2, random_state=0).fit_transform(normalize_data)
# X_TSNEprojected = TSNE(n_components=2, random_state=0).fit_transform(normalize_data)
x1 = X_TSNEprojected[:, 0]
x2 = X_TSNEprojected[:, 1]

plt.scatter(x1, x2, c=label, alpha=0.5, cmap="rainbow")
plt.colorbar()
plt.savefig('result.png')

dataset = x1.flatten()
th = 10
pred = np.array(list(map(lambda x: is_zero(th, x), dataset)))
label = np.array(list(map(lambda x: "zero" if x == 0 else "other", label)))

print(f"正解率: {accuracy_score(label, pred )*100}%")