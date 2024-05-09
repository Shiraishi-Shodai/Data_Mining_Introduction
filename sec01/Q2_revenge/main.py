import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib

from sklearn.metrics import  mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score

import tensorflow as tf

import torch
import torch.nn as nn

df = pd.read_csv('jkrt.csv')

# 欠損値の確認
print("欠損値の確認")
print(df.isnull().sum())
# 列情報
print("データフレームの情報を表示")
print(df.info())
# 重複した行の確認
print("重複した行の確認")
print(df.duplicated().sum())
# 元データのプロット
# plt.plot(df["x"], df["y"], c="orange")
# plt.savefig("original.png")

X_train = df.query("x > 0 and x < 5")["x"].to_numpy().reshape(-1, 1)
y_train = df.query("x > 0 and x < 5")["y"].to_numpy().reshape(-1, 1)
X_test = df.query("x >= 5")["x"].to_numpy().reshape(-1, 1)
y_test = df.query("x >= 5")["y"].to_numpy().reshape(-1, 1)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, input_shape=(1,), activation='relu'),
    tf.keras.layers.Dense(15, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1),
])

model.compile(optimizer="sgd", loss="mse", metrics=["mse"])
model.summary()

tf.keras.utils.plot_model(model)
history = model.fit(X_train, y_train, epochs=100, batch_size=5, validation_data=(X_test, y_test))
loss = model.evaluate(X_test, y_test)
print(f"MSE: {loss}")
result = pd.DataFrame(history.history)

# result[['loss', 'val_loss']].plot()
# plt.savefig('result1.png')
# result[['mse', 'val_mse']].plot()
# plt.savefig('result2.png')
y_pred = model.predict(X_test)
plt.plot(X_test, y_pred)
plt.plot(df["x"], df["y"], c="orange")
plt.savefig('test.png')
