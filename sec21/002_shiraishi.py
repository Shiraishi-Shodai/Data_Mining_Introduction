import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import math
import japanize_matplotlib
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.preprocessing import PolynomialFeatures


"""
このデータを多項式の曲線で近似してみよう。最小二乗誤差が10.5 以下になるようにしたい。最低でも何次の多項式で近似する必要があるか。また、その時の各係数の値を求めなさい。
"""

df = pd.read_csv('exe21.csv')
x = df["x"].to_numpy().reshape(-1, 1)
y = df["y"].to_numpy()
th = 10.5
ans_deg = None

np.set_printoptions(suppress=True, precision=3)

for deg in range(1, 21):
    
    polynomial_features = PolynomialFeatures(degree=deg)
    poly_x = polynomial_features.fit_transform(x)
    model = LinearRegression()
    model.fit(poly_x, y)
    y_pred = model.predict(poly_x)
    plt.title(f"y = x * {model.coef_[0]:.3f} + {model.coef_[1]:.3f} * sin(2π * x) + {model.intercept_:.3f}")
    plt.plot(x, y_pred, c="red")
    plt.scatter(x, y)
    plt.xlabel("x")
    plt.ylabel("y")
    # plt.savefig('predict2.png')
    # plt.show()
    mse = math.floor(np.mean((y - y_pred) ** 2) * 10) / 10
    print(f'平均二乗誤差 {mse}')
    
    if(int(mse *  10) <= int(th * 10)):
        ans_deg = deg
        print(model.coef_)
        print(model.intercept_)
        break

print(f'最小二乗誤差が{th}以下になるようにしたい。最低でも{deg}次の多項式で近似する必要がある')
