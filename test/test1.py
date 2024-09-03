from matplotlib import pyplot as plt
import japanize_matplotlib
import pandas as pd
import numpy as np

def main():
    df = pd.read_csv("test1.csv")
    plt.scatter(df["Ad_Spend"], df["Sales"])
    plt.title("広告費用と売上高")
    plt.xlabel("広告費用")
    plt.ylabel("売上高")
    plt.show()

    X = df["Ad_Spend"].to_numpy()
    y = df["Sales"].to_numpy()
    z = np.log10(y)
    plt.scatter(X, z)
    plt.title("広告費用と売上高")
    plt.xlabel("広告費用の対数")
    plt.ylabel("売上高")
    plt.show()


if __name__ == "__main__":
    main()