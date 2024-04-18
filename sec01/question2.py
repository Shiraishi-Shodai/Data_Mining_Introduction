# 2024年4月18日の演習問題

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import japanize_matplotlib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans

# ちょっと難しい回帰の問題を解いてみましょう。このデータはある関数のデータです。（ノイズあり）0<x<5の範囲でyを学習して、5≦xの範囲のyを予測してみてください。またその予測した値y_predと実際のyのデータはどれくくらい正確か評価してみてください。

df = pd.read_csv("apple_quality.csv")
# print(df.isnull().sum())
# print(type(df.head()))

df = df.drop("A_id", axis = 1)
df_y = df["Quality"].copy() # コピーしておかないと後の処理で消えちゃう
df_x = df.drop("Quality", axis = 1)
print(df.columns)
# X = df_x.to_numpy() #X=df_x.values は古い！
# y = df_y.to_numpy()
# X_train = X[0:1200,:] #0行目から1199行目までの全部の列（:）
# y_train = y[0:1200]
# print(X_train)
# print(y_train)