import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.metrics import PredictionErrorDisplay
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

## データを呼び出す
df = pd.read_csv("reg_data.csv")
x = df["x"].to_numpy().copy()
t = df["t"].to_numpy().copy()

## 単回帰分析
x = x.reshape(-1,1) #列ベクトルにする。 
model = LinearRegression()
model.fit(x,t)

# 本当の値t と予測値y_predを比較する
t_pred = model.predict(x)
#display = PredictionErrorDisplay(y_true=t, y_pred=t_pred)
#display.plot()

plt.scatter(t,t-t_pred)
plt.xlabel("目的変数t")
plt.ylabel("残差")
plt.title("演習問題15の残差プロット")
plt.show()

