import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import math
import japanize_matplotlib
import pandas as pd
from scipy.optimize import curve_fit

"""
周期は山と山の間隔を測るとほぼ 1 である。よって、これは１次関数と正弦曲線の合成であると予測して(※下記参照）

関数 y = Ax + Bsin(2πx) + C　

で近似できると仮定する。
この線形基底回帰モデルを各係数 A, B, C を決定することで定めなさい

"""

def to_vector(x):
    return np.array([x, np.sin(2 * 3.14 * x), 1]).reshape(1, -1)

def polyno(vec):
    poly_x = to_vector(vec[0]).reshape(1, -1)
    for k in range(1, len(vec)):
        poly_x = np.append(poly_x, to_vector(vec[k]), axis=0)
    return poly_x


df = pd.read_csv('exe21.csv')
x = df["x"].to_numpy()
y = df["y"].to_numpy()

plt.scatter(x, y)
plt.xlabel("x")
plt.ylabel("y")
plt.savefig('original.png')

# 自作関数バージョン
poly_x = polyno(x)
model = LinearRegression()
model.fit(poly_x, y)
y_pred = model.predict(poly_x)
print(model.coef_)
print(model.intercept_)
print(f'A={model.coef_[0]} B={model.coef_[1]} C={model.intercept_}')
plt.title(f"y = x * {model.coef_[0]:.3f} + {model.coef_[1]:.3f} * sin(2π * x) + {model.intercept_:.3f}")
plt.plot(x, y_pred, c="red")

# sickit-learnバージョン
# def func(x, a1, a2, a3):
#     return a1*x + a2 * np.sin(2 * 3.14 * x) + a3
# popt, pcov = curve_fit(func, x, y, p0=(1.0, 2.0, 3))
# a1 = popt[0]
# a2 = popt[1]
# a3 = popt[2]
# print(f"A={a1},B={a2},C={a3}")
# x_line = np.linspace(0, 6, 100)
# y_pred = func(x_line, a1, a2, a3)
# plt.plot(x_line, y_pred, c="red")

plt.scatter(x, y)
plt.xlabel("x")
plt.ylabel("y")
plt.savefig('predict.png')
