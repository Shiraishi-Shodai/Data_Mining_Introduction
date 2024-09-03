import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import math
import japanize_matplotlib
import pandas as pd

def to_vector(x):
    return np.array([1, math.sin(3.14 / 5 * x), x**2]).reshape(1, -1)

def glm_func(vec):
    # poly_xにデータを格納するためのリストを作る
    poly_x = to_vector(vec[0]).reshape(1, -1)

    for k in range(1, len(vec)):
        poly_x = np.append(poly_x, to_vector(vec[k]), axis=0)
    return poly_x

df = pd.read_csv("pol_test_sin.csv")
x = df["x"].to_numpy()#.reshape(-1, 1)
y = df["y"].to_numpy()

x_poly = glm_func(x)
model = LinearRegression()
model.fit(x_poly, y)
print(model.coef_)
print(model.intercept_)

y_pred = model.predict(x_poly)

plt.title("タイトル")
plt.scatter(x, y)
plt.scatter(x, y_pred, c="red")
plt.show()