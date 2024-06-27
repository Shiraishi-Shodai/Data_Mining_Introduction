import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import math
import japanize_matplotlib
import pandas as pd
from scipy.optimize import curve_fit

def func(x,a1,a2,a3):
    return a1*x + a2*np.sin(3.14/5*x)+a3*x**2

df = pd.read_csv("pol_test_sin.csv")
x= df["x"].to_numpy()
y= df["y"].to_numpy()

# 重みの初期値をp0の引数にタプルで渡す
popt,pcov = curve_fit(func,x,y,p0=(1,0.2,0.3))
a1 =popt[0]
a2 =popt[1]
a3 =popt[2]
print(f"a1={a1},a2={a2},a3={a3}")
x_line = np.linspace(0,20,100)
y_pred = func(x_line,a1,a2,a3)

plt.xlabel("x")
plt.ylabel("y")
plt.title("複数関数の線形結合による回帰")
plt.scatter(x, y)
plt.plot(x_line, y_pred, color='r')
plt.show()

