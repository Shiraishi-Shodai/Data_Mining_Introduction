"""
末廣くんのデータシャッフルコード
"""

import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.semi_supervised import LabelSpreading
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV

def interleave_data(X, y):
    # ラベル y の中のユニークなクラス（0, 1, 2）を取得
    classes = np.unique(y)
    # クラスごとのインデックスのリスト
    class_indices = [np.where(y == cls)[0] for cls in classes]
    # 各クラスのデータポイントの数を計算し、その中で最も少ないデータポイント数を取得(後にやる並べ替えの回数上限)
    min_len = min(len(indices) for indices in class_indices)
    # 各クラスのデータのインデックスを交互に並べる
    interleaved_indices = []
    for i in range(min_len):
        for indices in class_indices:
            interleaved_indices.append(indices[i])
    # 残りのデータを追加
    for indices in class_indices:
        interleaved_indices.extend(indices[min_len:])
    # 配列に変換
    interleaved_indices = np.array(interleaved_indices)
    # インデックスに基づいて X と y を並べ替え、並べ替えたデータを返す
    return X[interleaved_indices], y[interleaved_indices]


df = load_iris(as_frame=True).frame
X, y = df.iloc[:, :-1].to_numpy(), df["target"].to_numpy()

a, b = interleave_data(X, y)
# print(df.head())
# print(a[:5, :])
# print(b[:5])