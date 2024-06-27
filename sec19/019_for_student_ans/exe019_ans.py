

import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

"""
周期ωの正弦曲線 y = sin(2πx/ω) の0.5周期から2周期分のデータをノイズなしで作成。それを多項式で近似する。

"""
deg=4
omega = 2.0 #2*3.1415926535
A = 10
x = np.linspace(0.5*omega,2*omega,100).reshape(-1,1)
y = A*np.sin(2*3.1415926535/omega*x) +1.2*A

#以下ノイズ入れる
rng = np.random.default_rng()
noise = rng.uniform(-0.8, 0.8, (len(x), 1))
noise = 0 #この行をコメントアウトするとのノイズが乗る
#ここまで　ノイズ入れる
y = y + 1.1*noise

polynomial_features= PolynomialFeatures(degree=deg)
x_poly = polynomial_features.fit_transform(x)
model = LinearRegression()
model.fit(x_poly,y)
print(model.coef_)
print(model.intercept_)


for i in range(0,deg+1):
    if i==0:
        y_pred = model.intercept_[0]
    else:
        y_pred += model.coef_[0,i] *x**i
    
csv_array=np.concatenate([x,y.reshape(-1,1)],1)
print(csv_array)
np.savetxt("ploy_reg.csv",csv_array,delimiter=',',fmt='%.3f', header='x,y',footer='', comments='',)

#以下グラフの描画
plt.scatter(x,y,color="blue",label="観測データ")
plt.plot(x,y_pred,color="red",label="予測値")
plt.legend()
sin_func_str = f"y = {A:.1f}sin({2*3.1415926535/omega:.3f})x+{1.2*A:0.2f}"
plt.title(f"sin関数{sin_func_str}を多項式で近似する（次数={deg}）")
plt.xlabel("x")
plt.ylabel("y")
plt.savefig(f"sin_by_poly_{deg}.png")
plt.show()

print(np.mean((y-y_pred)**2))
