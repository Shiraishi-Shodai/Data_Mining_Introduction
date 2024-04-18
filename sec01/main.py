# 2024年4月11日の演習問題

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import japanize_matplotlib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans

if __name__ == "__main__":
    
    """
    Q1 
        この CSV を変数名 df というデータフレームに読み込みなさい。
    	ただし、A_id 列は削除すること。
       （目的:CSVファイルをデータフレームとして読み込み、各種の関数で処理できるか。）
    """
    
    df = pd.read_csv('apple_quality.csv')
    df = df.iloc[:, 1:]
    print(df.columns, end="\n\n")

    # null値の確認
    print(f"nullを確認 {df.isnull().sum()}")

    """
    Q5
        # Quality列の値は文字列であるが、2種の値になっているかを確認せよ。
        # また、全データについて、good は 1 に、badは 0 にラベルエンコーディ
        # ングしなさい。
    """

    df = df.replace({'Quality': {"good": 0, "bad": 1}})
    print(df["Quality"].unique())

    """
    Q2 
        Size 列,  Weight 列の最小値、最大値、平均値を    
    """
    
    view_columns = ["Size", "Weight"]
    for column in view_columns:
        
        min_value = df[column].min()
        max_value = df[column].max()
        mean_value = df[column].mean()

        print(f"{column}列の最小値 {min_value}")
        print(f"{column}列の最大値 {max_value}")
        print(f"{column}列の平均 {mean_value}\n")

    """
    Q3 
        データフレーム df の先頭 1200 行のデータを取り出し、NumPy配列にすることを考える。
        Quality列を y_trainに、それ以外の7つの列を X_trainというNumpy配列にしなさい。
    """
    
    # 特徴量を取得
    X = df.iloc[:, 0:-1].to_numpy()
    # 目的変数を取得
    y = df["Quality"].to_numpy()

    # 特徴量を標準化
    std_scaler = StandardScaler()
    X = std_scaler.fit_transform(X)

    X_train, X_test, y_train,  y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
    # print(X_train.shape)
    # print(X_test.shape)
    # print(type(X_test))

    """
    Q4 
        （3）の 1200 行のデータを用い、横軸に Size を、縦軸に Weight を
        とって、散布図を描きなさい。（目的:Matplotlib でグラフで単純な散布
        図を描くことができるか。）
    """

    plt.scatter(X_test[:, 0], X_test[:, 1])
    plt.title("タイトル")
    plt.grid()
    plt.show()

    """
    Q6
        Size,  Weight,  Sweetness,  Crunchiness,  Juiciness,  Ripeness,
        Acidity の７パラメータから Quality を予測する分類モデルを作りなさ
        い。訓練データとして先頭の1200行のデータを使いなさい。その際、
        scikit-learn を使い、どんな学習モデルを使うかは各自で決めなさい。
    """
    
    model = SVC()
    
    #  ハイパーパラメータのチューニング
    param_grid = {
        'C': [0.1, 0.5, 1], #  正則化パラメータ。Cの値が大きいほどモデルが訓練データに適合しようとする
        "gamma": [0.5, 1, 5] # カーネル関数パラメータ。gammaの値が大きいほど決定境界がより複雑になり、訓練データに適合しようとする
        }
    best_model = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    best_model.fit(X_train, y_train)
    print(f"最適なハイパーパラメーター {best_model.best_params_}")
    print(f"最高スコア {best_model.best_score_}", end="\n\n")

    # 学習データに対する精度
    y_train_pred = best_model.predict(X_train)
    print('----学習データに対する精度----')
    print(f"正解率 {accuracy_score(y_train, y_train_pred)*100} %", end="\n\n")

    # テストデータに対する精度
    y_test_pred = best_model.predict(X_test)

    # コンヒュージョンマトリックスの計算
    cm = confusion_matrix(y_test, y_test_pred) 
    tn, fp, fn, tp = cm.flatten()
    # 再現率
    recall_score =  tp / (fn + tp)
    # 適合率
    precision_score = tp / (fp + tp)

    print('----テストデータに対する精度----')
    print(f"正解率 {accuracy_score(y_test, y_test_pred)*100} %")
    print(f"再現率 {recall_score} %")
    print(f"適合率 {precision_score*100} %")

    display_labels = df["Quality"].unique()
    # print(display_labels)
    # print(type(display_labels))

    # コンヒュージョンマトリックスを描画
    disp = ConfusionMatrixDisplay(cm, display_labels=display_labels)
    disp.plot(cmap=plt.cm.Reds)

    plt.show()
    
    """
    （8）Classのラベルを使わずに、教師なし学習でクラスタリングすると、ラベルの結果とどれくらいの差が出るか。
    """
    
    color_palette = np.array(["orange", "blue"])
    
    model = KMeans(n_clusters=2, random_state=0)
    model.fit(X)
    model.labels_ = [1 if x == 0  else 0 for x in model.labels_]
    
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(X[:, 0], X[:, 1], c=color_palette[model.labels_])
    ax1.set_title("教師なし学習")
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(X[:, 0], X[:, 1], c=color_palette[y])
    ax2.set_title("元データ")
    plt.show()
    print("----教師なし学習の正解率----")
    print(f"{accuracy_score(model.labels_, y) * 100} %")