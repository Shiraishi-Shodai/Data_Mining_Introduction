import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import numpy as np
from sklearn.linear_model import LinearRegression

p_data = pd.read_csv('b.csv')
X = p_data['time'].to_numpy()
y = p_data['number'].to_numpy()
z= np.log10(y) #10を底にする常用対数をとる。

model = LinearRegression()
model.fit(X.reshape(-1,1),z) #Xと対数をとった後のzで単回帰をする
print(model.coef_)#傾きのNumpy配列 #[0.30103]
print(model.intercept_)#切片（スカラー）#0.47712125471965994
# z = 0.301 * X + 0.477
#pred_y = 10**(0.301 * X + 0.477) #単回帰を使った予測値
pred_y = 10**(model.coef_[0] * X + model.intercept_) #単回帰を使った予測値

plt.title("バクテリアの繁殖") 
plt.xlabel('時間')
plt.ylabel('バクテリア個数')
#plt.ylabel('個数(対数メモリ)')
plt.scatter(X, y)
plt.plot(X,pred_y,color="red")
#plt.savefig("bacteria.png")
plt.savefig("bacteria_pred.png")
plt.show()
