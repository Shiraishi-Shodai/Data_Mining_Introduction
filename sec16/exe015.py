"""
演習問題015　様々な方法で単回帰の問題を解く。
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib

df = pd.read_csv("reg_data.csv")
plt.scatter(df["x"],df["t"])
plt.xlabel("x")
plt.ylabel("t")
plt.scatter(df["x"],df["t"])
plt.title("x-t")

x = df["x"].to_numpy().copy()
t = df["t"].to_numpy().copy()

"""
③SymPyで偏微分
④解析解を使う
⑤解析解（疑似逆行列）
⑥scikit-learn
⑦勾配降下法
"""
###############ここから埋めよう################
def sol03(x,t):
    #sympyで最少となるa,bを求める。人間が式を展開して偏微分してもとめるのと同じ手順
    import sympy
    sympy.var('a b')
    E = np.sum((t-(a*x+b))**2)
    sol = sympy.solve([sympy.diff(E,b), sympy.diff(E,a)], [a, b])
    return [sol[a],sol[b]]

def sol04(x,t):
    # 正規方程式の解析解の利用
    x_mean = np.mean(x) #この後何度も使うので計算は1回だけ済ませておく。
    t_mean = np.mean(t) #この後何度も使うので計算は1回だけ済ませておく。

    s_xy = np.sum((x-x_mean)*(t-t_mean))
    s_x2 = np.sum((x-x_mean)**2)

    a = s_xy/s_x2
    b = t_mean-a*x_mean
    return [a,b]

def sol05(x,t):
    #正規方程式の解　ムーアペンローズの疑似逆行列
    X = np.concatenate([np.ones(len(t)).reshape(-1,1),x.reshape(-1,1)],1)
    w = np.linalg.inv(X.T @ X) @ X.T @ t
    return[w[1],w[0]]

def sol06(x,t):
    # 6.scikit-learnの利用
    # 本来モジュールの呼び出しはファイル先頭でするべきですが、対応するモジュールをわかりやすくするため、ここに書いています。
    from sklearn.linear_model import LinearRegression
    x = x.reshape(-1,1) #列ベクトルにする。 
    model = LinearRegression()
    model.fit(x,t)
    return [model.coef_[0],model.intercept_] #model.coef_はnumpy配列なので、0番目の要素を指定してスカラーにする。

#損失関数
def Error_func(vec_w,x,t):
    b = vec_w[0]
    a = vec_w[1]
    #global x,t
    return np.sum((t-(a*x+b))**2)/len(x)

#関数の勾配を求める関数
def diff_func_p(f,vec_w,x,t,h=0.0001) :
    grad = np.zeros_like(vec_w)
    for i in range(len(vec_w)):
        vec_i_org = vec_w[i]
        vec_w[i] = vec_i_org + h
        fh1=f(vec_w,x,t)
        vec_w[i] = vec_i_org - h
        fh2=f(vec_w,x,t)
        grad[i] = (fh1-fh2)/(2*h)
        vec_w[i] = vec_i_org
    return grad    

def sol07(x,t,max_iter = 1000,eta=0.005,vec_0= [4,4.0],print_screen=0):
    vec_w = np.array(vec_0) #重みの初期値

    for epoc in range(1,max_iter):
        grad = diff_func_p(Error_func,vec_w,x,t,0.000001)
        vec_w = vec_w - eta * grad
        if print_screen:
            print(str(epoc) +" th train : a=w_1= "+str(vec_w[0]) + ' , b=w_2 = '+str(vec_w[1]),end=" ")
            print("E="+str(Error_func(vec_w,x,t)) + " , grad= " +str(grad) )
        
        # 勾配がほぼ0になったら止める
        if (grad[0]**2 +grad[1]**2)**0.5 < 0.00001:
            break
        
    return [vec_w[1],vec_w[0]]

##############################################
a3,b3 = sol03(x,t)
a4,b4 = sol04(x,t)
a5,b5 = sol05(x,t)
a6,b6 = sol06(x,t)
a7,b7 = sol07(x,t,vec_0= [1.3,4.0],print_screen=0,max_iter = 3000) #print_screen=1 にすると学習の様子がわかる

print(f"3.の方法で解いたa={a3:.6f},b={b3:.6f}")
print(f"4.の方法で解いたa={a4:.6f},b={b4:.6f}")
print(f"5.の方法で解いたa={a5:.6f},b={b5:.6f}")
print(f"6.の方法で解いたa={a6:.6f},b={b6:.6f}")
print(f"7.の方法で解いたa={a7:.6f},b={b7:.6f}")

plt.plot(df["x"],a3*df["x"]+b3,color="red")
plt.savefig("x_t.png")
plt.show()