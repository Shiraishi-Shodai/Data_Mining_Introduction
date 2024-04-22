# 2024年4月18日の演習問題

import polars as pl
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib

from sklearn.metrics import  mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso

# 独自モジュールのインポート
from draw_data import DrawData
from pred import Pred
from data_preprocessor import DataPreprocessor

# ちょっと難しい回帰の問題を解いてみましょう。このデータはある関数のデータです。（ノイズあり）0<x<5の範囲でyを学習して、5≦xの範囲のyを予測してみてください。またその予測した値y_predと実際のyのデータはどれくくらい正確か評価してみてください。

if __name__ == "__main__":
  """データの読み込み"""
  df = pl.read_csv('jkrt.csv')
  # xが5より大きいデータはTrueとし、そうでないデータはFalseとするy_over_5列を追加
  df = df.with_columns(pl.Series(name="y_over_5", values=map(lambda x : True if x >= 5 else False, df["x"])))
  
  # """元データの可視化"""
  dd = DrawData()
  dd.show_null(df) # 欠損値の確認
  # 前処理前の学習データと検証データを用意
  df_train = df.filter(pl.col("y_over_5") == False)
  df_valid = df.filter(pl.col("y_over_5") == True)
  dd.draw_pair_plot(df, df_train, df_valid, original=True) # 学習データ、検証データのヒストグラムと散布図のペアプロットを描画

  # """前処理"""
  dp = DataPreprocessor()
  # データの水増し
  noise_df = dp.make_noise(df)  
  # マハラノビス距離を求め、ハズレ値を除外
  threshold = 2 # ハズレ値のしきい値
  mahala_df = dp.mahala(noise_df, threshold)
  # Xとyを標準化
  scaled_df = dp.scaling(mahala_df)
  machining_df = scaled_df
  
  """加工後データの可視化"""
  # 回帰のための前処理後の学習データと検証データを用意
  df_train = machining_df.filter(pl.col("y_over_5") == False)
  df_valid = machining_df.filter(pl.col("y_over_5") == True)
  dd.show_null(machining_df) # 欠損値の確認
  dd.draw_pair_plot(machining_df, df_train, df_valid, original=False) # 学習データ、検証データのヒストグラムと散布図のペアプロットを描画
  
  """データの予測"""
  pr = Pred()
  pr.tuning(machining_df)
  pr.show_result(df_train=df_train, df_valid=df_valid)
