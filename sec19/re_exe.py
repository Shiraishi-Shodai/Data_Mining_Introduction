import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd

df = pd.read_csv('ploy_reg.csv')
x = df["x"].to_numpy().reshape(-1, 1)
y = df["y"].to_numpy()


"""
パターン1
"""
# x_4 = x ** 4
# x_3 = x ** 3
# x_2 = x ** 2
# x_1 = x
# x_0 = x ** 0

# X = np.concatenate([x_0, x_1, x_2, x_3, x_4], 1)
# w = np.linalg.inv(X.T @ X) @ X.T @ y

# x_line = np.linspace(1, 4, 100)
# y_pred = w[0] + w[1] * x_line + w[2] * x_line**2 + w[3] * x_line**3 + w[4] * x_line**4

"""
パターン2
"""
degree = 6
polynomial_features = PolynomialFeatures(degree=degree)
x_poly = polynomial_features.fit_transform(x)
# print(x_poly.shape)

model = LinearRegression()
model.fit(x_poly, y)
y_pred = model.predict(x_poly)

print(model.coef_)
print(model.intercept_)
print(np.mean((y - y_pred)**2))

plt.scatter(x, y, c="g")
# plt.scatter(x_line, y_pred, c="red")
plt.scatter(x, y_pred, c="red")
plt.xlabel('x')
plt.ylabel('y')
plt.title("ploy_reg")
plt.show()
