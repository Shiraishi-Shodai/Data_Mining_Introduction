import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, PredictionErrorDisplay

data = pd.read_csv("Transistor_count.csv")
# print(data.isnull().sum())
# print(data.describe())

x = data['year'].to_numpy().reshape(-1, 1)
t = data["MOS_transistor_count"].to_numpy()

# 外れ値の除去


# """（３）z = log10(y) として、x から z を予測する式を求めよ。"""
z = np.log10(t) # 10を底とする常用対数を取る

"""
外れ値を削除する。その前にxとzに相関関係があるか調べどの「距離」を使って
外れ値を判定するか判断したい。

外れ値を判断するために使用する「距離」の候補。
① ユークリッド距離
② マハラノビス距離

相関係数が0.5以上だとマナラノビス距離を使用し
それ以下だとユークリッド距離を使用する。
"""

# xとzの相関係数を取る
corrcoef = np.corrcoef([x.flatten(), z])[0, 1]
print(f"xとzの相関係数は: {corrcoef}") # 9761272428841768

"""
xとzには0.5以上の相関関係があるので、ここではマハラノビス距離を使ってみる
"""
t_mu = t.mean()
t_std = t.std()

t_mahala = np.array([abs(tn - t_mu) / t_std for tn in t]) 

q0, q25, q50, q75, q100 = np.percentile(t_mahala, q = [0, 25, 50, 75, 100], method="midpoint")

iqr = q75 - q25                # 四分位範囲
lower_fence = q25 - 1.5 * iqr  # 下限の外れ値
upper_fence = q75 + 1.5 * iqr  # 上限の外れ値

use_index = [tn for tn in t_mahala if tn > lower_fence and tn < upper_fence ]

# model = LinearRegression()
# model.fit(x, z)
# """（４）x から y を予測する式を求めよ。"""
# t_pred = 10 ** (model.coef_[0] * x + model.intercept_)
# z_pred = model.coef_[0] * x + model.intercept_

# """（６）x, z の回帰で R2 決定係数を求めよ。"""
# print(f'決定係数 {r2_score(t, t_pred)*100}')
# print(f'傾き: {model.coef_[0]}')
# print(f'切片: {model.intercept_}')

# """（５）残差を残差プロットに表せ。（x, z の回帰で）"""
# # display = PredictionErrorDisplay(y_true=z, y_pred=z_pred.flatten())
# # display.plot()
# # plt.savefig("residual_plot.png")

# """（１）xy 平面にこのデータを散布図としてプロットせよ。"""
# plt.plot(x, t)
# """（２）次に、縦軸を log10(y)（常用対数）としてプロットしてみよ。"""
# plt.scatter(x, z)
# plt.plot(x, t_pred, c='red')
plt.title('ムーアの法則')
plt.xlabel('年')
plt.ylabel('集積回路あたりの部品数')

# # # plt.savefig('original_moore.png')
# plt.savefig('x_z_scatter.png')
# # plt.savefig('linear_pred.png')
# plt.show()

# """（７）この結果から、ムーアの法則（yは、2年ごとに2倍になる）は正しいといえるかどうかあなたの考えを述べなさい。"""