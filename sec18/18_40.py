import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import pandas as pd
import math

df = pd.read_csv("ling.csv")
x= df["x"].to_numpy().reshape(-1,1)
y= df["y"].to_numpy()

plt.xlabel("x")
plt.ylabel("y")
plt.title("元データ")
plt.scatter(x, y,c="g")
plt.show()

x_0 = x**0
x_1 = x
x_2 = x**2
x_3 = x**3
X = np.concatenate([x_0,x_1,x_2,x_3],1)
print(X)
w = np.linalg.inv(X.T @ X) @ X.T @ y
print(w.T)

x_line = np.linspace(0,20,100)
y_pred = w[0] + w[1]*x_line + w[2]*x_line**2 + w[3]*x_line**3

plt.xlabel("x")
plt.ylabel("y")
plt.title("元データ")
plt.plot(x_line, y_pred,c="r")
plt.scatter(x, y,c="g")
plt.show()
