import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd

df = pd.read_csv("sam01.csv")
x= df["x"].to_numpy().reshape(-1,1)
y= df["y"].to_numpy()
"""
自作の多項式用関数を作り、回帰する。

"""
def to_vector(x,n):
    return np.array([x**i for i in range(0,n+1)]).reshape(1,-1)

# vecの各要素に対してto_vector(x,n)してlen(vec)行n列の行列(Ndarray)を作る
def polyno(vec,n):
    poly_x = to_vector(vec[0],n).reshape(1,-1)
    for k in range(1,len(vec)):
        poly_x = np.append(poly_x, to_vector(vec[k],n), axis=0)
    return poly_x

polynomial_features= PolynomialFeatures(degree=3)
x_poly = polynomial_features.fit_transform(x)
#x_poly = polyno(x,3)
model = LinearRegression()
#model = Lasso(alpha=1.5)
model.fit(x_poly, y)
y_pred = model.predict(x_poly)

print(model.coef_)
print(model.intercept_)

plt.xlabel("x")
plt.xlabel("y")
plt.title("多項式による回帰のためのデータ")
plt.scatter(x, y)
plt.plot(x, y_pred, color='r')
plt.show()

