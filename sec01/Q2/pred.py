import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib
import numpy as np
import polars as pl

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
          "alpha" : [0.01, 0.1, 0.2, 0.3, 0.4, 1, 1.5, 3, 5, 10, 20, 30, 100, 100],
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
      ax.text(0.01, 0.75, f'決定係数: {r2_score(y, y_pred) * 100 :.2f} %', transform=ax.transAxes)
      ax.scatter(X, y, label="実測値", c="orange")
      ax.plot(X, y_pred, label="予測値", c="blue")
      ax.legend(bbox_to_anchor=(0, 1), loc='upper left', borderaxespad=0)
      ax.set_title(f'{key}データに対する予測値')

    def show_result(self, machining_df_train, machining_df_valid,):
        """学習データと検証データに対する予測結果を表示
        """
              
        self.plot_pred_dict["train"] = machining_df_train
        self.plot_pred_dict["valid"] = machining_df_valid

        fig = plt.figure()
        for index, item in enumerate(self.plot_pred_dict.items()):
            key, val = item[0], item[1]
            ax = fig.add_subplot(1, 2, index + 1)
            X = val["x"].to_numpy().reshape(-1, 1) # X_train["x"] or X_valid["x"]
            y = val["y"].to_numpy().reshape(-1, 1) # y_train["y"] or y_valid["y"]
            self.pred_process(key, X, y, ax)
            plt.grid()
            plt.savefig(f'pred_data.png')
    
    def gbm(self, df_train, df_valid):
      X_train = df_train["x"].to_numpy().reshape(-1, 1)
      y_train = df_train["y"].to_numpy()
      X_valid = df_valid["x"].to_numpy().reshape(-1, 1)
      y_valid = df_valid["y"].to_numpy()
      train_data = lgb.Dataset(X_train, y_train)
      valid_data = lgb.Dataset(X_valid, y_valid)

      params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'verbose': 2,
      }
      
      gbm = lgb.train(
        params,
        train_data,
        valid_sets=valid_data,
        num_boost_round=1000
      )
      
      train_pred = gbm.predict(X_train)
      print('学習データに対する予測')
      print(f'決定係数: {r2_score(y_train, train_pred)*100}% ')
      print(f'平均二乗誤差: {mean_squared_error(y_train, train_pred)}')
      valid_pred = gbm.predict(X_valid)
      print('テストデータに対する予測')
      print(f'決定係数: {r2_score(y_valid, valid_pred)*100}% ')
      print(f'平均二乗誤差: {mean_squared_error(y_valid, valid_pred)}')
      
      fig = plt.figure()
      ax1 = fig.add_subplot(1, 2, 1)
      ax2 = fig.add_subplot(1, 2, 1)
      ax1.text(0.01, 0.80, f'MSE {mean_squared_error(y_train, train_pred):.2f}', transform=ax1.transAxes)
      ax1.scatter(X_train, y_train, label="実測値", c="yellow")
      ax1.plot(X_train, train_pred, label="予測値", c="red")
      ax1.legend(bbox_to_anchor=(0, 1), loc='upper left', borderaxespad=0)
      ax1.set_title(f'trainデータに対する予測値')
      ax2.text(0.01, 0.80, f'MSE {mean_squared_error(y_valid, valid_pred):.2f}', transform=ax2.transAxes)
      ax2.scatter(X_valid, y_valid, label="実測値", c="yellow")
      ax2.plot(X_valid, valid_pred, label="予測値", c="red")
      ax2.legend(bbox_to_anchor=(0, 1), loc='upper left', borderaxespad=0)
      ax2.set_title(f'validデータに対する予測値')
      