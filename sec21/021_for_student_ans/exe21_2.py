import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import math
import japanize_matplotlib
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

def to_vector(x):
    #近似する関数のスタイルは　関数 y = Ax + Bsin(2πx) + C　
    return np.array([x ,math.sin(2*3.14*x),1]).reshape(1,-1)

# vecの各要素に対してto_vector(x,n)してlen(vec)行n列の行列(Ndarray)を作る
def polyno(vec):
    poly_x = to_vector(vec[0]).reshape(1,-1)
    for k in range(1,len(vec)):
        poly_x = np.append(poly_x, to_vector(vec[k]), axis=0)
    return poly_x

def get_mse(deg):
    df = pd.read_csv("exe21.csv")
    x= df["x"].to_numpy()
    y= df["y"].to_numpy()#.reshape(-1,1)
    polynomial_features= PolynomialFeatures(degree=deg)
    #x_poly = polynomial_features.fit_transform(x)
    x_poly = polyno(x) #変換後のX

    model = LinearRegression()
    #model.fit(x_poly, y)
    model.fit(x_poly[0:61], y[0:61])
    #print(model.coef_)
    #print(model.intercept_)

    y_pred = model.predict(x_poly)
    #print(f"平均二乗誤差＝{np.mean((y-y_pred)**2)}")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"{deg}次の多項式による近似曲線")
    plt.scatter(x, y)
    plt.plot(x[0:61], y_pred[0:61] ,color='b')
    plt.plot(x[61:], y_pred[61:], color='r')
    plt.savefig(f"./predict_curve_byfunc.png")
    plt.close()
    return [model.coef_,model.intercept_,np.mean((y-y_pred)**2)]

print(get_mse(0))
"""
by cahatgpt
A≈1.556
𝐵≈9.865
𝐶≈1.998

"""
