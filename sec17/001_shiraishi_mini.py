import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data = pd.read_csv('reg_data.csv')
x = data["x"].to_numpy().reshape(-1, 1)
t = data["t"].to_numpy()

model = LinearRegression()
model.fit(x, t)
y_pred = model.predict(x)

print(f'決定係数は {r2_score(t, y_pred)* 100}%です')

