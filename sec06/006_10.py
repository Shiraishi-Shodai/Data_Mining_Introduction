import numpy as np
import matplotlib.pyplot as plt
from skfuzzy.cluster import cmeans
import pandas as pd

df = pd.read_csv('iris.csv',usecols=['sepal_length','sepal_width','petal_length','petal_width'])
X = df.values
# print(X.T)
def target_to_color(target):
    if type(target) == np.ndarray:
        return (target[0], target[1], target[2]) # rgb
    else:
        print("rgb"[target])
        return "rgb"[target]

m = 5
c_means = cmeans(X.T, 3, m, 0.003, 10000)
print(c_means[1])
plt.figure()
plt.scatter(X[:,1], X[:,2], c=[target_to_color(t) for t in c_means[1].T])
plt.xlabel('sepal_width')
plt.ylabel('petal_length')
#plt.savefig('c_means.png')
plt.show()

# print(type(c_means))
# print(X.shape)
# print(c_means[0]) # 1行分のデータが各クラスタの座標(ここでは4次元の特徴量が存在するため4列になる)
# print(c_means[1]) # 各値のクラスタへの帰属度


# print(c_means)