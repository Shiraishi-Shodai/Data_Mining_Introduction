import numpy as np
import polars as pl

import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib

from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    def __init__(self) -> None:
        self.ss = StandardScaler()
    
    def mahala(self, df, threshold):
        """① 外れ値の除去(xとyには相関関係が存在するためマハラノビス距離を使用)
        Args :
            df: dataframe
        Return:
            df: ハズレ値を削除したデータフレーム
        """
        
        vec_x = df.to_numpy()
        vec_mu = np.mean(vec_x, axis=0)
        S = np.cov(vec_x.T)
        mahala_list = [distance.mahalanobis(x, vec_mu, np.linalg.pinv(S)) for x in vec_x]
        # マハラノビス距離をデータフレームに追加
        df = df.with_columns(pl.Series(name="mahala", values=mahala_list)) 
        # マハラノビス距離を描画
        plt.figure()
        plt.title("xy mahala")
        plt.plot(mahala_list, c="r")
        plt.grid()
        plt.savefig("mahala_graph.png")

        mahala_df = df.filter(pl.col('mahala') < threshold) # ハズレ値を省く
        mahala_df = mahala_df.drop("mahala")
        return mahala_df

    def scaling(self, df):
        """②標準化
        Args: 
            df: 標準化対象
        Return:
            scaled_df 標準化後データ
        """
        
        scaled_df = df
        # Xを標準化
        X = self.ss.fit_transform(scaled_df["x"].to_numpy().reshape(-1, 1))
        X = np.ravel(X)
        scaled_df = scaled_df.with_columns(pl.Series(name="x", values=X))
        # yを標準化
        y = self.ss.fit_transform(scaled_df["y"].to_numpy().reshape(-1, 1))
        y = np.ravel(y)
        scaled_df = scaled_df.with_columns(pl.Series(name="y", values=y))
        
        return scaled_df

    def make_noise(self, df):
        """③ノイズを追加したデータの追加
        Args: 
            df: ノイズを追加する対象のデータフレーム
        Return: 
            noise_df: ノイズを追加したデータフレーム"""
        
        # ノイズデータフレーム
        noise_df = df

        for i in range(7): # 学習データを80% テストデータを20%の配分にしたい

            mean = 0
            std = 1
            X_plus_noise = df["x"] + np.random.normal(mean, std, df["x"].shape)
            X_minus_noise = df["x"] - np.random.normal(mean, std, df["x"].shape)
            y_plus_noise = df["y"] + np.random.normal(mean, std, df["y"].shape)
            y_minus_noise = df["y"] - np.random.normal(mean, std, df["y"].shape)
            
            X_noise = pl.concat([X_plus_noise, X_minus_noise])
            y_noise = pl.concat([y_plus_noise, y_minus_noise])
            
            # 生成したノイズを一時的に格納するデータフレーム
            tmp_df = pl.DataFrame({
                "x": y_noise,
                "y": X_noise
            })
            
            # # 元のデータフレームとノイズデータフレームを結合
            noise_df = pl.concat([noise_df, tmp_df])
                
        return noise_df