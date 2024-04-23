# 2024年4月18日の演習問題

import polars as pl
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib

from sklearn.metrics import  mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score

# 独自モジュールのインポート
from draw_data import DrawData
from pred import Pred
from data_preprocessor import DataPreprocessor

# ちょっと難しい回帰の問題を解いてみましょう。このデータはある関数のデータです。（ノイズあり）0<x<5の範囲でyを学習して、5≦xの範囲のyを予測してみてください。またその予測した値y_predと実際のyのデータはどれくくらい正確か評価してみてください。

def machining(df):
    """データフレームを加工し、返却する
    Args:
        df: 前処理対象のデータフレーム
    
    Return:
        df: 前処理後のデータフレーム
    """
    dp = DataPreprocessor()
    # ノイズデータを加えてデータの水増し
    noise_df = dp.make_noise(df)  
    # # マハラノビス距離を求め、ハズレ値を除外
    threshold = 1.5 # ハズレ値のしきい値
    mahala_df = dp.mahala(noise_df, threshold)
    # Xとyを標準化
    scaled_df = dp.scaling(mahala_df)
    
    return scaled_df
  
if __name__ == "__main__":
  """データの読み込み"""
  df = pl.read_csv('jkrt.csv')
  
  """元データの可視化"""
  dd = DrawData()
  dd.show_null(df) # 欠損値の確認
  df_train = df.filter((pl.col('x') > 0) & (pl.col('x') < 5))
  df_valid = df.filter(pl.col("x") > 5)
  dd.draw_pair_plot(df, df_train, df_valid, original=True) # 学習データ、検証データのヒストグラムと散布図のペアプロットを描画
  
  """学習とテストデータに対して前処理"""
  machining_df_train = machining(df_train)
  machining_df_valid = machining(df_valid)
  machining_df = pl.concat([machining_df_train, machining_df_valid])
  
  """加工後データの可視化"""
  dd.show_null(machining_df) # 欠損値の確認
  dd.draw_pair_plot(machining_df, machining_df_train, machining_df_valid, original=False) # 学習データ、検証データのヒストグラムと散布図のペアプロットを描画
  
  """データの予測"""
  pr = Pred()
#   pr.tuning(machining_df)
#   pr.show_result(machining_df_train, machining_df_valid,)
  pr.gbm(machining_df_train, machining_df_valid,) # lightGBMを使った回帰