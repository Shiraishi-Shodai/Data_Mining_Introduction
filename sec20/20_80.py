import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import math
import japanize_matplotlib
"""
データを作る
"""
x = np.linspace(0.1,20,50)
y_true = 12+ 10*np.sin(3.14/5*x)+0.1*x**2 
y = y_true + np.random.uniform(-1.8,1.8,len(x))
csv_array = np.concatenate([x.reshape(-1,1),y.reshape(-1,1)],1)
np.savetxt("pol_test.csv",csv_array, delimiter=",", header="x,y", comments='')
plt.xlabel("x")
plt.ylabel("y")
plt.title("多項式による回帰のためのデータ")
plt.scatter(x, y)
plt.plot(x, y_true, color='g')
plt.show()

