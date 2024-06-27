import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd

df = pd.read_csv("sam01.csv")
x= df["x"].to_numpy().reshape(-1,1)
y= df["y"].to_numpy()

polynomial_features= PolynomialFeatures(degree=3) # degreeは何次式か
x_poly = polynomial_features.fit_transform(x) # 変換スライド18の21ページ(行数: データ数, 列数: N次式)
print(x_poly)
model = LinearRegression()
# model = Lasso(alpha=1.5)

model.fit(x_poly, y)
y_pred = model.predict(x_poly)
print(model.coef_)
print(model.intercept_)

plt.xlabel("x")
plt.ylabel("y")
plt.title("多項式による回帰のためのデータ")
plt.scatter(x, y)
plt.plot(x, y_pred, color='r')
plt.show()

