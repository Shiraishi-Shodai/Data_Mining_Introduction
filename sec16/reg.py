import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import japanize_matplotlib
from sklearn.metrics import r2_score
from sklearn.metrics import PredictionErrorDisplay

df = pd.read_csv('reg_data.csv')
X = df["x"].to_numpy().reshape(-1, 1)
t = df["t"].to_numpy()

X2 = np.concatenate([np.ones(X.shape), X], axis=1)

b, a = np.linalg.inv(X2.T @ X2) @ X2.T @ t

y_pred = X * a + b
y_pred = y_pred.flatten()

r2_score1 = 1 - (sum((t - y_pred)**2) / sum((t - t.mean())**2))
r2_score2 = r2_score(t, y_pred)
# print(f"決定係数: {r2_score1}")
print(f"決定係数: {r2_score2}")

display = PredictionErrorDisplay(y_true=t, y_pred=y_pred)
display.plot()

# plt.title('x-t')
# plt.scatter(X.flatten(), t)
# plt.plot(X.flatten(), y_pred)
plt.show()