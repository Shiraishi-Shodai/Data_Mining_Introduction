import sympy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.linear_model import LinearRegression

"""
SymPyを使って求める
"""
def sol03(X, t):
    sympy.var('a, b')
    
    E = 0 # 最小二条誤差を求める式
    
    for xi, ti in zip(X, t):
        E += (ti - (a * xi + b))**2 
    
    # print(sympy.expand(E))
    
    # 偏微分を求める
    a_diff = sympy.diff(E, a)
    b_diff = sympy.diff(E, b)

    # 連立方程式を解く
    ans = sympy.solve([a_diff, b_diff], [a, b])
    return ans[a], ans[b]    

"""
解析解を使って求める
"""
def sol04(X, t):
    x_mean = X.mean()
    t_mean = t.mean()
    Sxy = sum((X - x_mean) * (t - t_mean))
    Sx2 = sum((X - x_mean) ** 2)
    
    a = Sxy / Sx2
    b = t_mean - a * x_mean
    
    return a, b

"""
疑似逆行列を使って求める
"""
def sol05(X, t):
    X = np.concatenate((np.ones(X.shape[0]).reshape(-1, 1), X.reshape(-1, 1)), axis=1)
    b, a = np.linalg.inv(X.T @ X) @ X.T @ t
    return a, b


"""
scikit-learnを使う
"""
def sol06(X, t):
    model = LinearRegression()
    model.fit(X.reshape(-1, 1), t)
    
    return model.coef_[0], model.intercept_

def main():
    df = pd.read_csv('reg_data.csv')
    X = df["x"].to_numpy()
    t = df["t"].to_numpy()

    print(df.isnull().sum())
    print(df.describe(),)
    print()
    
    a3, b3 = sol03(X, t)
    print(f"{'SymPy':<20}: a={a3}, b={b3}")
    
    a4, b4 = sol04(X, t)
    print(f"{'解析解':<20}: a={a4}, b={b4}")
    
    a5, b5 = sol05(X, t)
    print(f"{'疑似逆行列':<20}: a={a5}, b={b5}")

    a6, b6 = sol06(X, t) 
    print(f"{'LinearRegression':<20}: a={a6}, b={b6}")

if __name__ == "__main__":
    main()