import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, PredictionErrorDisplay
import math

data = pd.read_csv('ch.csv')
x = data["x"].to_numpy().reshape(-1, 1)
y = data["y"].to_numpy()

# 65.0の項を消す
y -= 65
z = np.log10(y)

model = LinearRegression()
model.fit(x, z)
z_pred = model.predict(x)
y_pred = 10 ** (z_pred)

# display = PredictionErrorDisplay(y_true=y, y_pred=y_pred)
# plt.title('残差プロット')
# display.plot()
# plt.savefig('002_residual_plot.png')

P = 300
"""
① 300 = 10 ** (model.coef_[0] * x + model.intercept_)
② 両辺の対数を取り、xを指数から取り出す
    np.log10(300) = model.coef_[0] * x + model.intercept_
③ 移項する
    (np.log10(P) - model.intercept_) / model.coef_[0]
"""

x300_pred = (np.log10(P) - model.intercept_) / model.coef_[0]

print(f'パラメータA {model.coef_[0]}')
print(f'パラメータB {model.intercept_}')
print(f'決定係数 {r2_score(z, z_pred)}')
print(f'血中成分が{P}になるのは接種後{math.floor(x300_pred * 2) / 2}時間後')

plt.scatter(x, z)
# plt.scatter(x, z)
# plt.plot(x, y_pred, c='red')
plt.title('血中成分量')
# plt.title('血中成分量の常用対数')
plt.xlabel('時間')
plt.ylabel('血中成分')
# plt.savefig('002_original.png')
# plt.savefig('002_result.png')
plt.savefig('test2.png')