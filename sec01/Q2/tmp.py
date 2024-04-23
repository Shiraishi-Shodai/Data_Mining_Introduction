import polars as pl

# 例として、2つのSeriesを作成する
series1 = pl.Series("A", [1, 2, 3])
series2 = pl.Series("B", [4, 5, 6])

# Seriesを縦方向に結合する
concatenated_series = pl.concat([series1, series2])

# 結果を表示する
print(concatenated_series)
