from matplotlib import pyplot as plt
import japanize_matplotlib
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

"""
（１）xy 平面にこのデータを散布図としてプロットせよ。
（２）次に、縦軸を log10(y)（常用対数）としてプロットしてみよ。
（３）z = log10(y) として、x から z を予測する式を求めよ。
（４）x から y を予測する式を求めよ。
（５）残差を残差プロットに表せ。（x, z の回帰で）
（６）x, z の回帰で R2 決定係数を求めよ。
（７）この結果から、ムーアの法則（yは、2年ごとに2倍になる）は正しいといえるかどうかあなたの考えを述べなさい。
"""

def before_treatment(df: pd.DataFrame):
    # 欠損値の確認
    print(df.isnull().sum())
    # 基本情報
    print(df.describe())
    # 年ごとのデータの個数を表示
    print(df["year"].value_counts())
    X = df["year"].to_numpy()
    y = df["MOS_transistor_count"].to_numpy()
    # plt.scatter(X, y)
    # plt.title("ムーアの法則")
    # plt.xlabel("年")
    # plt.ylabel("トランジスタ数")
    # plt.show()
    # plt.savefig("origin.png")

def pretreatment(df: pd.DataFrame) -> pd.DataFrame:
  mean_df = df.groupby("year")["MOS_transistor_count"].mean().reset_index()
#   print(mean_df)
#   print(mean_df.shape)

  return mean_df

def main():
    df = pd.read_csv('Transistor_count.csv')

    # before_treatment(df)
    mean_df = pretreatment(df)
    X = mean_df["year"].to_numpy()
    y = mean_df["MOS_transistor_count"].to_numpy()

    # 予測
    # 10を底としてときのyの対数
    z = np.log10(y)
    plt.scatter(X, z)
    plt.title("ムーアの法則")
    plt.xlabel("年")
    plt.ylabel("10を底としてときのyの対数")
    plt.show()

    model = LinearRegression()
    model.fit(X.reshape(-1, 1), z)
    a = model.coef_[0]
    b = model.intercept_
    y_pred = 10**(a * X + b)
    plt.scatter(X, y, c="red")
    plt.plot(X, y_pred, c="blue")
    plt.title("ムーアの法則")
    plt.xlabel("年")
    plt.ylabel("トランジスタ数")
    plt.xlim(X.min(), X.max() + 10)
    plt.show()

    # 残差プロット
    plt.title("残差プロット")
    plt.xlabel("年")
    plt.ylabel("残差")
    z_pred = model.predict(X.reshape(-1, 1))
    plt.scatter(X, z - z_pred)
    plt.show()

    print(f"決定係数: {r2_score(z, z_pred)*100}%")

if __name__ == "__main__":
    main()
