import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import RandomizedSearchCV

class Pred:
    def __init__(self) -> None:
        self.plot_pred_dict = {
            "train": None,
            "valid": None,
            }
        
        self.model = None
        
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
        param_dic = {
          "alpha" : [0.1, 1.0, 5.0],
          "solver" : ["auto", "svd", "cholesky"],
          "max_iter" : [1000, 2000]
        }
        model = Ridge()
        clf = RandomizedSearchCV(model, param_dic, random_state=0)
        clf.fit(X, y)
        print(f'ベストパラメータ {clf.best_params_}')
        print(f'ベストスコア {clf.best_score_}')
        
        self.model = clf

    def pred_process(self, key, X, y, ax):
      """与えられたx, yに対する予測をし、平均二乗誤差を求め、回帰直線を描画
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
      y_pred = self.model.predict(X)
      print(f'平均二乗誤差 {mean_squared_error(y, y_pred)}')

    # 既存のテキスト要素があれば削除する(テキストを生成すると、デフォルトの位置がAxesの右下になるから。2つめのグラフのラベルが右下に来てしまう)
      for text in ax.texts:
        text.remove()
        
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