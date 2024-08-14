from matplotlib import pyplot as plt
import japanize_matplotlib
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv('b.csv')
X = df["time"].to_numpy()
y = df["number"].to_numpy()

z = np.log10(y)
model = LinearRegression()
model.fit(X.reshape(-1, 1), z)
y_pred = 10 ** (model.coef_[0] * X + model.intercept_)

plt.title('バクテリアの繁殖')
plt.xlabel('時間')
plt.scatter(X, y, label="元データ")
plt.plot(X, y_pred, c="red", label="予測データ")
plt.legend()
plt.grid()
plt.show()