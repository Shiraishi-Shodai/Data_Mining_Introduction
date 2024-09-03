import numpy as np
import sympy
from sklearn.linear_model import LinearRegression

def sol03(data_ex):
    E = 0
    sympy.var("a, b")
    for li in data_ex:
        E += (li[1] - (a * li[0] + b))**2

    # print(sympy.expand(E))
    # print(sympy.diff(E, a))
    # print(sympy.diff(E, b))

    res = sympy.solve([sympy.diff(E, a), sympy.diff(E, b)], [a, b])
    print(E.subs([(a, res[a]), (b, res[b])]))

def sol04(data_ex):
    x = data_ex[:, 0]
    y = data_ex[:, 1]

    Sxy = sum(((x - x.mean()) * (y - y.mean()))) / len(x)
    Sx2 = sum((x - x.mean())**2) / len(x)

    a = Sxy / Sx2
    b = y.mean() - a * x.mean()
    print(a, b)

def sol05(data_ex):
    x = data_ex[:, 0].reshape(-1, 1)
    y = data_ex[:, 1]
    x = np.concatenate([np.ones(len(x)).reshape(-1, 1), x], axis=1)
    
    t = data_ex[:, 1].copy()
    w = np.linalg.inv(x.T @ x) @ x.T @ t
    print(w)


def sol06(data_ex):
    x = data_ex[:, 0].reshape(-1, 1)
    y = data_ex[:, 1]
    model = LinearRegression()
    model.fit(x, y)
    print(model.coef_, model.intercept_)

def main():
    data_ex =np.array([
        [ 1.0 , 3.1 ] ,
        [ 2.0 , 5.1 ] ,
        [ 3.0 , 6.8 ] ,
        [ 4.0 , 8.9 ] ,
        [ 5.0 , 11.5 ] ,
        [ 6.0 , 13.8 ] ,
    ])

    # sol03(data_ex)
    # sol04(data_ex)
    sol05(data_ex)
    # sol06(data_ex)

if __name__ == "__main__":
    main()