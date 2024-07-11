"""
プロンプト
このCSVファイルのデータはx,yの2つの列からなるデータです。
yをxの関数で近似したいと思います。A,B,Cを定数として
y = Ax + Bsin(2πx) + C で近似するとき、元のデータとの
誤差が最小となるようなA,B,Cを決定してください。pythonのコードも提示してください。

"""
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd

# Load data
data = pd.read_csv('exe21.csv')
x_data = data['x']
y_data = data['y']

# Define the function to fit
def func(x, A, B, C):
    return A * x + B * np.sin(2 * np.pi * x) + C

# Fit the function to the data
params, params_covariance = curve_fit(func, x_data, y_data, p0=[1, 1, 1])

# Extract the fitting parameters
A, B, C = params

# Print the results
print(f"A = {A}, B = {B}, C = {C}")

# Plot the original data and the fitted function
plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data, label='Data')
plt.plot(x_data, func(x_data, A, B, C), color='red', label='Fitted function')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Data and Fitted Function')
plt.savefig("fit_curve.png")
plt.show()
