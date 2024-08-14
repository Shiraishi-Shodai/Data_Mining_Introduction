from matplotlib import pyplot as plt
import japanize_matplotlib
import pandas as pd
import numpy as np

df = pd.read_csv('b.csv')
X = df["time"].to_numpy()
y = df["number"].to_numpy()

plt.title('バクテリアの繁殖')
plt.xlabel('時間')
plt.scatter(X, y)
plt.show()