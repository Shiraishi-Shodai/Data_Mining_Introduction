import numpy as np

class NyLinearRegression2:
    
    def __init__(self):
        self.coef_ =False
        self.intercept_ =False

    def predict(self,X):
        # Xに対してNumpy配列かどうかのチェックを入れる。
        predict = X * self.coef_ + self.intercept_
        return predict
    
    # 学習する。    
    def fit(self,X,y,**kwargs):
        #任意の関数を微分する
        def diff_func_p(f,vec_w,h=0.0001) :
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


        #損失関数 1引数でないとだめ
        def Error_func(vec_w):
            a = vec_w[0]
            b = vec_w[1]            
            E=0
            #for i in range(0,len(X)):
            #    E +=(y[i]-(a*X[i]+b))**2
            E = np.sum((y-(a*X+b))**2)
            return E
        
        #vec_w=np.array([2.0,3.0])
        vec_w = kwargs['w_init']
        eta = kwargs['eta'] #0.002
        iteratr = kwargs['iteratr'] #1000
        for epoc in range(1,iteratr):
            grad = diff_func_p(Error_func,vec_w,0.000001)
            vec_w = vec_w - eta * grad
        
        #得られたインスタンス変数をセットする
        self.coef_ =vec_w[0]
        selfintercept_ =vec_w[1] 
        return vec_w
    
    #正解は当然100％
    def score(self,X,y):
        #何をもって正解かの定義をする。

        return 1.0


