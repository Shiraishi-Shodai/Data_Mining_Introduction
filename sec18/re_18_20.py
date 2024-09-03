import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd

df = pd.read_csv("sam01.csv")
X = df["x"].to_numpy().reshape(-1, 1)
y = df["y"].to_numpy()

# plt.scatter(X, y)
# plt.show()

x_0 = X**0
x_1 = X
x_2 = X**2
x_3 = X**3

X = np.concatenate([x_0, x_1, x_2, x_3], 1)

x_line = np.linspace(0, 20, 100)
# polynomial_features = PolynomialFeatures(degree=2)
# x_poly = polynomial_features.fit_transform(X)
# print(x_poly)
# model = LinearRegression()
# model.fit(x_poly, y)