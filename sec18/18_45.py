import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd

df = pd.read_csv("ling.csv")
x= df["x"].to_numpy().reshape(-1,1)
y= df["y"].to_numpy()

polynomial_features= PolynomialFeatures(degree=3)
x_poly = polynomial_features.fit_transform(x)
model = LinearRegression()

model.fit(x_poly, y)
print(model.coef_)
print(model.intercept_)
