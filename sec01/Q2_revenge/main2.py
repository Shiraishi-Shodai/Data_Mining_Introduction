import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib

from sklearn.metrics import  mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score
from sklearn.kernel_approximation import RBFSampler
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import KFold, cross_val_score

df = pd.read_csv('jkrt.csv')

# 欠損値の確認
print("欠損値の確認")
print(df.isnull().sum())
# 列情報
print("データフレームの情報を表示")
print(df.info())
# 重複した行の確認
print("重複した行の確認")
print(df.duplicated().sum())
# 元データの描画
# plt.plot(df["x"], df["y"], c="orange", label="元データのプロット")
# plt.scatter(df["x"], df["y"], c="blue", label="元データの散布図")
# plt.legend()
# plt.savefig("original.png")

X = df["x"].to_numpy().reshape(-1, 1)
y = df["y"].to_numpy()
X_train = df.query("x > 0 and x < 5")["x"].to_numpy().reshape(-1, 1)
y_train = df.query("x > 0 and x < 5")["y"].to_numpy()
X_test = df.query("x >= 5")["x"].to_numpy().reshape(-1, 1)
y_test = df.query("x >= 5")["y"].to_numpy()

model = KernelRidge(alpha=0.0002, kernel='rbf')
cv = KFold(n_splits=5, random_state=0, shuffle=True)
scores = cross_val_score(model, X_train, y_train, scoring='r2', cv=cv, n_jobs=1)
print(f'決定係数の平均値 {scores.mean()}')

model.fit(X, y)
y_pred = model.predict(X)

plt.plot(X, y_pred, c='blue')
plt.plot(X, y, c="orange")
plt.savefig('pred.png')

print(f'決定係数 { r2_score(y, y_pred)*100} % ')
print(f'MSE {mean_squared_error(y, y_pred)}')