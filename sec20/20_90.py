import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import math
import japanize_matplotlib
import pandas as pd

def to_vector(x):
    return np.array([x ,math.sin(3.14/5*x),x**2]).reshape(1,-1)

# vecの各要素に対してto_vector(x,n)してlen(vec)行n列の行列(Ndarray)を作る
def polyno(vec):
    poly_x = to_vector(vec[0]).reshape(1,-1)
    for k in range(1,len(vec)):
        poly_x = np.append(poly_x, to_vector(vec[k]), axis=0)
    return poly_x


df = pd.read_csv("pol_test_sin.csv")
x= df["x"].to_numpy()#.reshape(1,1)
y= df["y"].to_numpy()

x_poly = polyno(x)
model = LinearRegression()
model.fit(x_poly, y)
print(model.coef_)
print(model.intercept_)

y_pred = model.predict(x_poly)
plt.xlabel("x")
plt.ylabel("y")
plt.title("複数関数の線形結合よる回帰")
plt.scatter(x, y)
plt.plot(x, y_pred, color='r')
plt.show()

