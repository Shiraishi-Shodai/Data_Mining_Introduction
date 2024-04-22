# 2024年4月18日の演習問題

import polars as pl
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso

from scipy import stats
from scipy.spatial import distance


# ちょっと難しい回帰の問題を解いてみましょう。このデータはある関数のデータです。（ノイズあり）0<x<5の範囲でyを学習して、5≦xの範囲のyを予測してみてください。またその予測した値y_predと実際のyのデータはどれくくらい正確か評価してみてください。
"""データの読み込み"""
df = pl.read_csv('jkrt.csv')

"""元データの可視化"""
# 欠損値の確認
print(df.null_count())

def df_train_test_split(df):
  """データフレームを学習データとテストデータに分割
  
  Args : DataFrame
  Return: Train DataFrame, Valid DataFrame
  """
  
  df_train = df.filter((pl.col("x") > 0) & (pl.col("x") < 5))
  # print(df_train.shape) (24, 2)
  df_valid = df.filter(pl.col('x') >= 5)

  return df_train, df_valid

# 前処理前の学習データと検証データを用意
df_train, df_valid = df_train_test_split(df)

plot_scatter_dict = {
  "original" : df,
  "train" : df_train,
  "valid" : df_valid,
}

# 学習データ、検証データのヒストグラムと散布図のペアプロットを描画
fig = plt.figure()
for index, item in enumerate(plot_scatter_dict.items()):
  
  key, val = item[0], item[1]
  ax = fig.add_subplot(1, 3, index + 1)
  ax.set_title(key)
  
  x = val["x"].to_numpy()
  y = val["y"].to_numpy()
  xy_covariance = np.cov(x, y)
  xy_corrcoef = np.corrcoef(x, y)[0, 1]
  ax.text(0.01, 0.95, f"X variance {xy_covariance[0, 0]:.2f}", transform = ax.transAxes, c="red")
  ax.text(0.01, 0.90, f"y variance {xy_covariance[1, 1]:.2f}", transform = ax.transAxes, c="red")
  ax.text(0.01, 0.85, f"Xy covariance {xy_covariance[0, 1]:.2f}", transform = ax.transAxes, c="red")
  ax.text(0.01, 0.80, f"X corrcoef {xy_corrcoef:.2f}", transform = ax.transAxes, c="red")
  
  sns.jointplot(x="x", y="y", data=val, palette='tab10')
  plt.grid()
  plt.savefig(f"{key}_join_plot.png")

"""前処理"""
# 外れ値の計算(xとyには相関関係が存在するためマハラノビス距離を使用)
# vec_x = df.to_numpy()
# vec_mu = np.mean(vec_x, axis=0)
# S = np.cov(vec_x.T)
# mahala_list = [distance.mahalanobis(x, vec_mu, np.linalg.pinv(S)) for x in vec_x]
# # マハラノビス距離をデータフレームに追加
# df = df.with_columns(pl.Series(name="mahala", values=mahala_list)) 
# # マハラノビス距離を描画
# plt.figure()
# plt.title("xy mahala")
# plt.plot(mahala_list, c="r")
# plt.grid()
# plt.savefig("mahala_graph.png")

# threshold = 1.3 # ハズレ値のしきい値
# df = df.filter(pl.col('mahala') < threshold) # ハズレ値を省く

# ハイパーパラメーターのチューニングのためのXとyを用意
X = df["x"].to_numpy().reshape(-1, 1)
y = df["y"].to_numpy().reshape(-1, 1)

# Xを標準化
ss = StandardScaler()
# X = ss.fit_transform(X)
# y = ss.fit_transform(y)

# 回帰のための前処理後の学習データと検証データを用意
df_train, df_valid = df_train_test_split(df)

"""データの予測"""
# # ハイパーパラメータのチューニング
param_dic = {
  "alpha" : [0.1, 1.0, 5.0],
  "solver" : ["auto", "svd", "cholesky"],
  "max_iter" : [1000, 2000]
}
model = Ridge()
clf = RandomizedSearchCV(model, param_dic, random_state=0)
clf.fit(X, y)
print(f'ベストパラメータ {clf.best_params_}')
print(f'ベストスコア {clf.best_score_}')

def pred_process(key, X, y, ax):
  """与えられたx, yに対する予測をし、平均二乗誤差を求め、回帰直線を描画
  Args: 
    key: train or valid
    X       : X_train or X_valid
    y       : y_train or y_valid
    ax      : Axes
  
  Return:
    None
  """
  print(f'-------{key}データに対する予測-------')
  y_pred = clf.predict(X)
  print(y.shape, y_pred.shape)
  print(f'平均二乗誤差 {mean_squared_error(y, y_pred)}')

  ax.text(0.01, 0.95, f'MSE {mean_squared_error(y, y_pred):.2f}', transform=ax.transAxes)
  ax.scatter(X, y)
  ax.plot(X, y_pred)
  ax.set_title(f'{key}データに対する予測値')

plot_pred_dict = {
  "train": df_train,
  "valid": df_valid,
}

fig = plt.figure()
for index, item in enumerate(plot_pred_dict.items()):
  key, val = item[0], item[1]
  ax = fig.add_subplot(1, 2, index + 1)
  X = val["x"].to_numpy().reshape(-1, 1) # X_train or X_valid
  y = val["y"].to_numpy().reshape(-1, 1) # y_train or y_valid
  pred_process(key, X, y, ax)
  plt.savefig(f'pred_data.png')
