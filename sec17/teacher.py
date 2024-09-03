import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import japanize_matplotlib
# data is from https://en.wikipedia.org/wiki/Transistor_count#Microprocessors
df = pd.read_csv('Transistor_count.csv')

X = df["year"].to_numpy()
y =df["MOS_transistor_count"].to_numpy()
log10y =np.log10(y)
X = X.reshape(-1,1)
#print(X)
#print(y)

## (1)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_title("回帰直線")
ax.set_xlabel('発表年')
ax.set_ylabel('トランジスタ数')
ax.scatter(X, y, c='b',label="トランジスタ数")
ax.legend()
plt.show()

## (2)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_title("回帰直線")
ax.set_xlabel('発表年')
ax.set_ylabel('トランジスタ数（対数目盛）')
ax.scatter(X, log10y, c='b',label="トランジスタ数")
ax.legend()
plt.show()

## (3)
model = LinearRegression()
model.fit(X, log10y)
print(f"求める式: z = log10(y) ={model.coef_[0]} * x  {model.intercept_}")

## (4)
print(f"求める式: y = 10**({model.coef_[0]} * x  {model.intercept_})")

## (5)
z_pred =  model.predict(X)
plt.scatter(log10y, z_pred - log10y, c = 'green', marker = 'o', label = '残差')
plt.title("残差プロット")
plt.xlabel('値[log10]')
plt.ylabel('残差[log10]')
plt.hlines(y = 0, xmin = 1, xmax = 12, lw = 2, color = 'red') # y = 0に直線を引く
plt.show()

##  (6)
from sklearn.metrics import r2_score
r2 = r2_score(log10y,z_pred)
print(f"決定係数R2={r2}")

## (7)
import math
print(f"log2(10**model.coef_[0]) = {math.log2(10**model.coef_[0])}はほぼ0.5である。")

log10y_pred = model.predict(X)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_title("回帰直線")
ax.set_xlabel('発表年')
ax.set_ylabel('トランジスタ数（対数目盛）')
ax.scatter(X, log10y, c='b',label="トランジスタ数")
ax.plot(X, log10y_pred, c='r',label="トランジスタ数予測値")
ax.legend()
plt.show()