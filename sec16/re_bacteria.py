from matplotlib import pyplot as plt
import japanize_matplotlib
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import PredictionErrorDisplay, r2_score

def main():
    df = pd.read_csv('b.csv')
    X = df["time"].to_numpy().reshape(-1, 1)
    y = df["number"].to_numpy()

    # plt.scatter(X, y)
    # plt.xlabel("時間")
    # plt.ylabel("バクテリアの数")
    # plt.title("バクテリアの増殖")
    # plt.show()

    z = np.log10(y)
    # plt.scatter(X, z)
    # plt.xlabel("時間")
    # plt.ylabel("10を底としたときのバクテリアの数の対数")
    # plt.title("バクテリアの増殖")
    # plt.show()

    model = LinearRegression()
    model.fit(X, z)
    z_pred = model.predict(X)
    # display = PredictionErrorDisplay(y_true=z, y_pred=z_pred)
    # display.plot()
    # plt.scatter(X, z - z_pred)
    # plt.xlabel("時間")
    # plt.ylabel("残差")
    # plt.title("残差プロット")
    # plt.show()

    y_pred = 10**z_pred
    plt.scatter(X, y)
    plt.plot(X, y_pred, c="red")
    plt.xlabel("時間")
    plt.ylabel("残差")
    plt.title("残差プロット")
    plt.show()

    print(f"傾き:{model.coef_[0]}")
    print(f"切片: {model.intercept_}")
    print(f"決定係数{r2_score(z, z_pred)*100}%")
    print(10**0.294)

if __name__ == "__main__":
    main()
