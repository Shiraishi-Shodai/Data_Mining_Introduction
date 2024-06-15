import numpy as np
class MyLinearRegression2:

    # コンストラクタ。インスタンス変数だけ格納する準備をしましょう。
    def __init__(self):
        self.coef_ =False
        self.intercept_ =False

    #偏微分する関数
    def diff_func_p(self,f,vec_w,h=0.0001) :
        grad = np.zeros_like(vec_w)
        for i in range(len(vec_w)):
            vec_i_org = vec_w[i]
            vec_w[i] = vec_i_org + h
            fh1=f(vec_w)
            vec_w[i] = vec_i_org - h
            fh2=f(vec_w)
            grad[i] = (fh1-fh2)/(2*h)
            vec_w[i] = vec_i_org
        return grad

    # numpy 配列で複数の説明変数をうけとり、それぞれに対応した予測値を返す
    def predict(self,X):
        a = self.coef_
        b = self.intercept_
        return a*X +b #ブロードキャスト
    
    def Error_func(self,x,y):
        E=0
        z = self.data_Y -(x*self.data_X + y)
        E = np.sum(z*z)
        return E

    # 学習はすなわち係数をセットすること  
    def fit(self,x_vec,y_vec):
        self.data_X = x_vec
        self.data_Y = y_vec
        eta = 0.002 #学習係数
        vec_w=np.array([3.1,1.0]) #重みの初期値

        for epoc in range(1,3000):
            grad = self.diff_func_p(Error_func,vec_w,0.000001)
            vec_w = vec_w - eta * grad
        self.coef_ = vec_w[0]
        self.intercept_ = vec_w[1]
        #一応各係数を返す
        return(np.array([aself.coef_,self.intercept_]))   

    #正解
    def score(self,X,y):
        return 1.0

