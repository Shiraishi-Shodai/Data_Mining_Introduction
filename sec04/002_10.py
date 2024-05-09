from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib

df = pd.read_csv('group_sample.csv')
X = df[['x1','x2']].to_numpy()
y_label = df['y'].to_numpy()

plt.scatter(X[:,0],X[:,1],color="navy")
plt.xlabel('x1')
plt.ylabel('x2')
plt.title("サンプルデータ")
plt.show()