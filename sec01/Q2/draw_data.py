import polars as pl
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib

class DrawData:
    def __init__(self) -> None:
        self.plot_scatter_dict = {
        "full" : None,
        "train" : None,
        "valid" : None,
        }
    
    def show_null(self, df):
        """ 欠損値の確認
        Args:
            df: データフレーム
        Return:
            None
        """
        print(df.null_count())

    def draw_pair_plot(self, df, df_train, df_valid, original=True):
        """学習データ、検証データのヒストグラムと散布図のペアプロットを描画
        Args:
            original:
            df: 
        
        Return:
            None
        """
        
        fnhead = "" # 描画するグラフのファイル名の先頭文字
        
        if original:
            fnhead = "original"
        else:
            fnhead = "machining"
            
        self.plot_scatter_dict["full"] = df
        self.plot_scatter_dict["train"] = df_train
        self.plot_scatter_dict["valid"] = df_valid
        fig = plt.figure()
        
        for index, item in enumerate(self.plot_scatter_dict.items()):
        
            key, val = item[0], item[1]
            ax = fig.add_subplot(1, 3, index + 1)
            ax.set_title(key)
            
            x = val["x"].to_numpy()
            y = val["y"].to_numpy()
            xy_covariance = np.cov(x, y)
            xy_corrcoef = np.corrcoef(x, y)[0, 1]
            ax.text(0.01, 0.95, f"X variance {xy_covariance[0, 0]:.2f}", transform = ax.transAxes, c="red")
            ax.text(0.01, 0.90, f"y variance {xy_covariance[1, 1]:.2f}", transform = ax.transAxes, c="red")
            ax.text(0.01, 0.85, f"Xy covariance {xy_covariance[0, 1]:.2f}", transform = ax.transAxes, c="red")
            ax.text(0.01, 0.80, f"X corrcoef {xy_corrcoef:.2f}", transform = ax.transAxes, c="red")
            
            sns.jointplot(x="x", y="y", data=val, palette='tab10')
            plt.grid()
            plt.savefig(f"{fnhead}_{key}_join_plot.png")