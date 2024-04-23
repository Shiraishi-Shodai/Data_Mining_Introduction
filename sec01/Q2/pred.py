import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib
import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import r2_score

import lightgbm as lgb

class Pred:
    def __init__(self) -> None:
        self.plot_pred_dict = {
            "train": None,
            "valid": None,
            }
        
        # self.model = lgb.LGBMRegressor(metric='rmse')
        self.model = Ridge(alpha=10, max_iter=10000)
        
        
    # ハイパーパラメータのチューニング
    def tuning(self, df):
        """ハイパーパラメータのチューニングを行いself.modelにチューニングしたモデルを代入
        Args:
            df: ランダムサーチCVに使用するデータを持つデータフレーム
        Return :
            None
        """
        X = df["x"].to_numpy().reshape(-1, 1)
        y = df["y"].to_numpy().reshape(-1, 1)
        
        # scikit-learn用のチューニング工程
        param_dic = {
          "alpha" : [0.01, 0.1, 0.2, 0.3, 0.4, 1, 1.5, 3, 5, 10, 20, 30, 100, 1000],
          "solver" : ["auto", "svd", "cholesky"],
          "max_iter" : [1000, 2000, 3000]
        }
        model = Ridge()
        clf = RandomizedSearchCV(model, param_dic, random_state=42, cv=5, n_iter=1000)
        # clf = GridSearchCV(model, param_dic, cv=5)
        clf.fit(X, y)
        print(f'ベストパラメータ {clf.best_params_}')
        print(f'ベストスコア {clf.best_score_}')
        self.model = clf
        

    def pred_process(self, key, X, y, ax):
      """与えられたx, yに対する予測をし、決定係数と平均二乗誤差を求め、回帰直線を描画
      Args: 
        key: train or valid
        X       : X_train or X_valid
        y       : y_train or y_valid
        ax      : Axes
        clf     : チューニング済みモデル
    
      Return:
        None
      """
      print(f'-------{key}データに対する予測-------')
      if key == "train":
        self.model.fit(X, y)
        
      y_pred = self.model.predict(X)
      
      y_mean = y.mean() 
      zenhendou = np.sum((y - y_mean)**2)
      kaikihendou = np.sum((y_pred - y_mean)**2)
      zansahendou = np.sum((y - y_pred)**2)
      print(f'全変動平方和: {(zenhendou)} ')
      print(f'回帰変動平方和: {(kaikihendou)} ')
      print(f'残差変動平方和: {(zansahendou)} ')
      print(f'決定係数: {r2_score(y, y_pred)*100}% ')
      print(f'平均二乗誤差: {mean_squared_error(y, y_pred)}')
      
      ax.text(0.01, 0.80, f'MSE {mean_squared_error(y, y_pred):.2f}', transform=ax.transAxes)
      ax.scatter(X, y, label="実測値", c="yellow")
      ax.plot(X, y_pred, label="予測値", c="red")
      ax.legend(bbox_to_anchor=(0, 1), loc='upper left', borderaxespad=0)
      ax.set_title(f'{key}データに対する予測値')

    def show_result(self, df_train, df_valid):
        """学習データと検証データに対する予測結果を表示
        """
              
        self.plot_pred_dict["train"] = df_train
        self.plot_pred_dict["valid"] = df_valid
        fig = plt.figure()
        for index, item in enumerate(self.plot_pred_dict.items()):
            key, val = item[0], item[1]
            ax = fig.add_subplot(1, 2, index + 1)
            X = val["x"].to_numpy().reshape(-1, 1) # X_train["x"] or X_valid["x"]
            y = val["y"].to_numpy().reshape(-1, 1) # y_train["y"] or y_valid["y"]
            self.pred_process(key, X, y, ax)
            plt.savefig(f'pred_data.png')