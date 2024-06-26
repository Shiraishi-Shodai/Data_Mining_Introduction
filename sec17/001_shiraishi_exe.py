import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, PredictionErrorDisplay
from scipy.spatial import distance

# デフォルトの表示設定を変更
pd.options.display.float_format = '{:.2f}'.format

def getDistance(x : np.ndarray, z: np.ndarray) -> np.ndarray:
    
    xz_distance = np.zeros(z.shape)
    # xとの相関係数を取る
    corrcoef = np.corrcoef([x.flatten(), z])[0, 1] * 100
    print(f"xとzの相関係数は: {corrcoef}")

    if int(corrcoef) >= 50:
        vec_x = np.concatenate([x, z.reshape(-1,1)], axis=1)
        vec_mu = np.mean(vec_x, axis=0) # 平均値のベクトルμを作る。
        S = np.cov(vec_x.T)             # 分散共分散行列 S を作る。

        # すべての点のマハラノビス距離リストを作る。
        xz_distance = np.array([distance.mahalanobis(x, vec_mu, np.linalg.pinv(S)) for x in vec_x])
    else:
        pass # ユークリッド距離は省略
    
    return xz_distance

data = pd.read_csv("Transistor_count.csv")
# print(data.isnull().sum())
# print(data.describe())

"""
年ごとの平均を取る(xとyを関数関係にする。xが決まればyが一意に決まる関係)
"""

mean_df = data.groupby(["year"]).mean()
mean_df = mean_df.reset_index()

x = mean_df['year'].to_numpy().reshape(-1, 1)
t = mean_df["MOS_transistor_count"].to_numpy()
# """（３）z = log10(y) として、x から z を予測する式を求めよ。"""
z = np.log10(t) # 10を底とする常用対数を取る

"""
外れ値を削除する。その前にxとzに相関関係があるか調べどの「距離」を使って
外れ値を判定するか判断したい。

外れ値を判断するために使用する「距離」の候補。
① ユークリッド距離
② マハラノビス距離

相関係数が0.5以上だとマハラノビス距離を使用し
それ以下だとユークリッド距離を使用する。
"""

xz_distance = getDistance(x, z)
q0, q25, q50, q75, q100 = np.percentile(xz_distance, q = [0, 25, 50, 75, 100], method="midpoint")

iqr = q75 - q25                # 四分位範囲
lower_fence = q25 - 1.5 * iqr  # 下限の外れ値
upper_fence = q75 + 1.5 * iqr  # 上限の外れ値

use_index = list(map(lambda n_distance: True if n_distance > lower_fence and n_distance < upper_fence else False, xz_distance))

x_use = x[use_index]
z_use = z[use_index]
t_use = t[use_index]

model = LinearRegression()
model.fit(x_use, z_use)
"""（４）x から y を予測する式を求めよ。"""
t_pred = 10 ** (model.coef_[0] * x_use + model.intercept_)
z_pred = model.coef_[0] * x_use + model.intercept_

# """（６）x, z の回帰で R2 決定係数を求めよ。"""
print(f'決定係数 {r2_score(z[use_index], z_pred)*100}')
# print(f'傾き: {model.coef_[0]}')
# print(f'切片: {model.intercept_}')
original_a = model.coef_[0]

# """（５）残差を残差プロットに表せ。（x, z の回帰で）"""
# display = PredictionErrorDisplay(y_true=z_use, y_pred=z_pred.flatten())
# display.plot()
# plt.title('残差プロット')
# plt.xlabel('年')
# plt.ylabel('残差')
# plt.savefig("residual_plot.png")

# """（１）xy 平面にこのデータを散布図としてプロットせよ。"""
# plt.scatter(x, t)
# """（２）次に、縦軸を log10(y)（常用対数）としてプロットしてみよ。"""
# plt.scatter(x_use, z_use)
# plt.plot(x_use, z_use)

# 最終結果
# plt.plot(x, t)
# plt.plot(x_use, t_pred, c='red')
# plt.title('元データ')
# plt.xlabel('年')
# plt.ylabel('集積回路あたりの部品数')
# plt.ylabel(' log10(t)（常用対数）')

# plt.savefig('original_data.png')
# plt.savefig('x_z_scatter2.png')
# plt.savefig('x_z_plot.png')
# plt.savefig('linear_pred.png')

# """（７）この結果から、ムーアの法則（yは、2年ごとに2倍になる）は正しいといえるかどうかあなたの考えを述べなさい。"""

"""
ムーアの予測どおりにトランジスタ数が増えた場合のデータを求め、元データと比較する
"""
t_start = t_use[0]
moore_arr = np.empty((0, 2))

for i in np.arange(min(x_use)[0], max(x_use)[0], 2):

    if moore_arr.size == 0: # データが空なら
        moore_arr = np.append(moore_arr, np.array([[i, t_start]]), axis=0)
    else:
        moore_arr = np.append(moore_arr, np.array([[i, moore_arr[moore_arr.shape[0] - 1, 1] * 2]]), axis=0)

moore_x = moore_arr[:, 0].reshape(-1, 1)
moore_t = moore_arr[:, 1]

plt.scatter(x, t, label='元データ')
plt.scatter(moore_x, moore_t, color='orange', label="ムーアの予測")

moore_z = np.log10(moore_t) # 10を底とする常用対数を取る
model.fit(moore_x, moore_z)
model.predict(moore_x)
moore_pred = 10 ** (model.coef_[0] * moore_x + model.intercept_)
print(f'傾き: {model.coef_[0]}')
print(f'切片: {model.intercept_}')
moore_a = model.coef_[0]

plt.plot(x, t, label="元データ")
plt.plot(moore_x, moore_pred, color='red', label='ムーアの予測')
plt.legend()
# plt.title('ムーアの予測と元データの比較1')
plt.title('ムーアの予測と元データの比較2')
plt.xlabel('年')
plt.ylabel('集積回路あたりの部品数')
# plt.savefig('result_scatter.png')
plt.savefig('result_plot.png')

print(f'オリジナルデータの傾き: {original_a}')
print(f'ムーアの予測の傾き: {moore_a}')

print(pd.DataFrame(moore_arr))
"""
ムーアの予測によって求めた指数の傾き0.150と元データから求めた指数の傾き0.146はだいたい同じくらいなのでムーアの予測は正しいと言って良いと考える
"""