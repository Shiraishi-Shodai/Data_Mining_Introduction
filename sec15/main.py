import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import japanize_matplotlib
import sympy
from sklearn.linear_model import LinearRegression
import torch
import torch.optim as optim

def plot(data: pd.DataFrame) -> None:
    plt.scatter(data["x"], data["t"])
    plt.xlabel("x")
    plt.ylabel("t")
    plt.title("オリジナルプロット")
    plt.savefig('original.png')

def result_view(sol: str, a: float, b: float)-> None:
    
    N = 7
    print(f"========={sol}========")
    print(f'aの値は {a}')
    print(f'bの値は {b}')
    print(f'N = 7のとき、答えは... {a * 7 + b}', end="\n\n")
    
def sol01(x: np.ndarray, t: np.ndarray) -> None:
    sol = "①全探索"
    a = 0.1
    a_min = a
    b = 0.1
    b_min = b
    h = 0.01 # aやbに加算する数
    E_min = 1000000 # 最小値
    E_current = 0
    
    while a < 10:
        while b < 10:
            E_current = sum((t - (x * a + b)) ** 2)
            
            if E_current < E_min:
                E_min = E_current
                a_min = a
                b_min = b
            b += h
        
        b = 0.1
        a += h
    
    result_view(sol, a_min, b_min)

def sol03(x: np.ndarray, t: np.ndarray)-> None:
    sol = "②SymPyで偏微分して求める"
    
    # a = sympy.Symbol('a')
    # b = sympy.Symbol('b')
    sympy.var('a b')
    E = sum((t - (a * x + b)) ** 2)
    diff_a = sympy.diff(E, a)
    diff_b = sympy.diff(E, b)
    # print(sympy.expand(E)) # 式を展開
    # print(diff_a) # aで偏微分
    # print(diff_b) # bで偏微分
    
    # 連立方程式
    res = sympy.solve([sympy.diff(E, a), sympy.diff(E, b)], [a, b])
    a_m = res[a]
    b_m = res[b]
    result_view(sol, a_m, b_m)
 
def sol05(x: np.ndarray, t: np.ndarray)-> None:
    prom = "⑤解析解を使う（疑似逆行列の利用）"
    x = np.concatenate([np.ones(len(x)).reshape(-1, 1), x.reshape(-1, 1)], axis=1)
    b, a = np.linalg.inv(x.T @ x) @ x.T @ t
    result_view(prom, a, b)

def sol06(x: np.ndarray, t: np.ndarray)-> None:
    sol = "⑥scikit-learnを使う"
    model = LinearRegression()
    model.fit(x.reshape(-1, 1), t)
    a = model.coef_[0]
    b = model.intercept_
    
    result_view(sol, a, b)

"""
7は以下の書籍より
PyTorch実践入門 ~ ディープラーニングの基礎から実装へ
https://www.amazon.co.jp/PyTorch%E5%AE%9F%E8%B7%B5%E5%85%A5%E9%96%80-Eli-Stevens/dp/4839974691/ref=asc_df_4839974691/?tag=jpgo-22&linkCode=df0&hvadid=342438969336&hvpos=&hvnetw=g&hvrand=6906019978293249133&hvpone=&hvptwo=&hvqmt=&hvdev=c&hvdvcmdl=&hvlocint=&hvlocphy=1009698&hvtargid=pla-1089258244212&psc=1&mcid=3161d7ade0d33334821f9255b1f3a9cc&th=1&psc=1
ゼロから作るDeep Learning
https://www.oreilly.co.jp/books/9784873117584/
"""
def sol07(x: np.ndarray, t: np.ndarray)-> None:
    prom = "⑦勾配降下法を使う"
    x: torch.Tensor = torch.from_numpy(x)
    t: torch.Tensor = torch.from_numpy(t)
    # params = torch.tensor([1.0, 0.0]) # 重みとバイアスを初期化
    params = torch.tensor([1.0, 0.0], requires_grad=True) # 重みとバイアスを初期化(requires_grad=Trueでテンソルのツリー全体を追跡することで誤差逆伝搬をしやすくしている？)
    lr = 1e-6 # 学習率
    epochs = 2000 # エポック数
    optimizer = optim.SGD(params=[params], lr=lr)
    
    def model(x, w, b) -> float:
        return w * x + b
    
    def loss_fn(t_pred, t) -> torch.Tensor:
        # 損失関数
        return ((t_pred - t) ** 2).mean()
    
    def dloss_fn(t_pred, t) -> torch.Tensor:
        # 損失関数の導関数
        return (2 * (t_pred - t)).mean()
    
    def dmodel_dw() -> torch.Tensor:
        # 最小二乗誤差の導関数をwで偏微分する
        return x
    
    def dmodel_db() -> float:
        # 最小二乗誤差の導関数をbで偏微分する
        return 1.0
    
    def grad_fn(x, w, b, t) -> torch.Tensor:
        t_pred = model(x, w, b)
        dloss_dtp = dloss_fn(t_pred, t)
        dloss_dw = dloss_dtp * dmodel_dw()
        dloss_db = dloss_dtp * dmodel_db()

        return torch.stack([dloss_dw.sum(), dloss_db.sum()])
    
    for epoch in range(1, epochs + 1):

        # if params.grad is not None:
        #     params.grad.zero_

        # 純伝搬
        t_pred = model(x, *params)
        loss = loss_fn(t_pred, t)

        # 逆伝搬
        # w, b = params
        # grad = grad_fn(x, w, b, t)
        # params -= lr * grad
        loss.backward()

        # パラメータの更新
        # with torch.no_grad():
        #     params -= lr * params.grad
        optimizer.step()
        
        # if epoch in [1, 10, 30, 50, 80, 100, 1000, 1500, 2000]:
        #     print(f'Epoch{epoch}の時、パラメータは{params[0], params[1]} 損失 : {loss}')

    a = float(params[0])
    b = float(params[1])
    result_view(prom, a, b)

def main() -> None:
    data = pd.read_csv('reg_data.csv')
    print(data.describe())
    print(data.isnull().sum())
    
    plot(data)
    
    x = data["x"].to_numpy()
    t = data["t"].to_numpy()
    
    sol01(x, t)
    sol03(x, t)
    sol05(x, t)
    sol06(x, t)
    sol07(x, t)
    
if __name__ == "__main__":
    main()
