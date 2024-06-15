import sympy
import numpy as np
from sklearn.linear_model import LinearRegression

data_ex=np.array([
    [1.0,3.1],
    [2.0,5.1],
    [3.0,6.8],
    [4.0,8.9],
    [5.0,11.5],
    [6.0,13.8],
])

model = LinearRegression()
model.fit(data_ex[:,0].reshape(-1, 1),data_ex[:,1])
print(model.coef_)
print(model.intercept_)
print(f"x=7の時の予測値は{model.predict(np.array([[7.0]]))}")

