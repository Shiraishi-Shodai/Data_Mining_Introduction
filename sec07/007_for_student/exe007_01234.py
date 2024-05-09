import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib

df = pd.read_csv('group_sample.csv')
X = df[['x1','x2']].to_numpy()
y_label = df['y'].to_numpy()
#print(X)
# 006_20.pyのDBSCANのソースコードを参考にして、人間のラベルと同じように3クラスタにできるか

plt.scatter(X[:,0], X[:,1],c=y_label)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('人間のラベルによる色分け')
plt.savefig('exe007_label.png')
plt.show()
