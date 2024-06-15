import numpy as np
class MyLinearRegression:

    # コンストラクタ。インスタンス変数だけ格納する準備をしましょう。
    def __init__(self):
        self.coef_ =False
        self.intercept_ =False

    # numpy 配列で複数の説明変数をうけとり、それぞれに対応した予測値を返す
    def predict(self,X):
        a = self.coef_
        b = self.intercept_
        return a*X +b #ブロードキャスト
    
    # 学習はすなわち係数をセットすること  
    def fit(self,x_vec,y_vec):
        #平均
        x_mu = sum(x_vec)/len(x_vec)
        y_mu = sum(y_vec)/len(y_vec)
        Sxy = 0
        Sxx = 0
        for i in range(0,len(x_vec)) :
            Sxy = (x_vec[i]-x_mu)*(y_vec[i]-y_mu) + Sxy
        
        for i in range(0,len(x_vec)) :
            Sxx = (x_vec[i]-x_mu)**2 + Sxx
        
        # a,b は1次回帰直線の傾きと係数
        a = Sxy / Sxx
        b = y_mu - a* x_mu
        
        self.coef_ = a
        self.intercept_ = b 
        #一応各係数を返す
        return(np.array([a,b]))   

    #正解
    def score(self,X,y):
        return 1.0

