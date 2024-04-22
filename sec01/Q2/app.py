import polars as pl

# データフレームを作成
df = pl.DataFrame({"x": [1, 2, 3, 4, 5]})

# 標準化スカラーを作成
scaler = pl.StandardScaler(columns=["x"])

# x列を標準化した値に置き換える
df = scaler.fit_transform(df)

# 結果
print(df)