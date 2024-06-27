import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import pandas as pd
import math
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

df = pd.read_csv('ploy_reg.csv')
x = df["x"].to_numpy().reshape(-1, 1)
y = df["y"].to_numpy()

# plt.scatter(x, y)
# plt.title('元データ')
# plt.xlabel("x")
# plt.ylabel("y")
# plt.savefig('19_original.png')

degree = 4

"""
"""

polynomial_features = PolynomialFeatures(degree=degree)
x_poly = polynomial_features.fit_transform(x)
# print(x_poly)

model = LinearRegression()
model.fit(x_poly, y)

y_pred = model.predict(x_poly)
mse = np.mean((y - y_pred)**2)

print(f'a1 a2 a3 a4: {model.coef_}')
print(f'a0 {model.intercept_}')
print(f'平均二条誤差 {mse}')

plt.scatter(x, y, label="観測データ")
plt.scatter(x, y_pred, c='red', label="予測値")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title(f'予測 (degree = {degree})')
plt.savefig(f'19_predict_{degree}degree.png')