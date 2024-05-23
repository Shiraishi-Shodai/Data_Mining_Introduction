"""
irisデータを読み込み、ラベルを次の割合でわざと消す。分類アルゴリズムSVMと半教師あり学習（ラベル拡散）を比較してそれぞれの正解率を求め、どちらが正解率が高いか調べよ。

（１）ラベルを全体の3％にしたとき

HalvingRandomSearchCV
初期段階でランダムにパラメータセットをサンプリングし、それぞれを試す。
各ラウンドごとに成績の悪いパラメータセットを切り捨て、残ったセットにより多くのリソースを割り当てまる。

factor : 各ラウンドで次に進むハイパーパラメータセットの数(上位何パーセントのパラメータセットを残すかを指定)
min_resources: 初期ラウンドで各ハイパーパラメータセットを評価する際に使用される最小リソース量
max_resources: 初期ラウンドで各ハイパーパラメータセットを評価する際に使用される最大リソース量

"""

import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.semi_supervised import LabelSpreading
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV

def shuffle(df, del_num):
    """
    Description: この後の教師あり学習や半教師なし学習で与える正解ラベルに偏りがないようデータをシャッフル
    
    Args:
        df(pd.DataFrame)        : load_irisで取得したデータフレーム
        del_num(int)            : ラベルを-1に変換する個数
        
    Return:
        shuffle_df(pd.DataFrame): 引数で受け取ったdfをシャッフルし、ラベルを-1にしない部分が均等に分割したデータフレーム
    """
    label_types = df["target"].unique()               # ラベルの種類
    len_df_indexes = df.shape[0]                      # dfの行数 
    tail_num = len_df_indexes - del_num               # 教師ラベルを残すデータの数
    min_num = tail_num // len(label_types)     # それぞれのラベルの種類に必要なデータの最低数
    
    print(f"教師データを残すラベルを等分してシャッフルするためにはそれぞれ{min_num}個必要です。")   
    
    
    flag = True  # while文を抜けるか判断するフラグ  
    shuffle_df = None
        
    while(flag):
        
        tmp_df = df.sample(frac=1.0)                      # dfのデータをすべてシャッフルしてサンプリング(fracはサンプリング数の割合を表す(0.0 ~ 1.0))
        tail_label = tmp_df["target"].tail(tail_num)     
    
        for i in label_types:
            label_count = (tail_label == i).sum()
            if min_num > label_count:
                break
            elif i == label_types[-1]:
                flag = False
                shuffle_df = tmp_df
    
    print(f"↓↓↓↓↓↓↓シャッフル後のデータフレームの後ろ5行のラベルカウント↓↓↓↓↓↓↓ \n {shuffle_df['target'].tail().value_counts()}")
    return shuffle_df
    
def make_label_color(label):
    """
    Description: 整数を受取りそれぞれの値に対応する文字列を返す
    
    Args: 
        label(int) : ラベルデータ(-1~2)
    
    Return:
        文字列(str): black,red, blue, greeのいずれかの文字列を返す
    """
    color_dict = { -1: "black", 0: "red", 1: "blue" }
    return color_dict.get(label, "green")

def label_machining(del_num, labels):
    """
    Description: 与えられたラベル配列を引数Nで指定された割合分、教師データとして残しそれ以外を-1とする。
    
    Args:
        del_num (int)                 : 教師データを削除する個数
        labels (ndarray dtype=int): 加工するラベル配列
        
    Return:
        machining_labels(ndarray dtype=int): 加工したラベル配列
    """
    
    machining_labels = labels.copy()
    
    machining_labels[0:del_num] = np.array(-1)
    print(f"-1のデータのラベルの割合 {(np.count_nonzero(machining_labels < 0)/ len(machining_labels) )*100}")
    print(f"-1以上のデータのラベルの割合 {np.count_nonzero(machining_labels >= 0) / len(machining_labels)*100}")
    
    return machining_labels

def plot(X1, X2, labels_color, file_name, title, xlabel="sepal width", ylabel="petal length"):
    """グラフを描画する(以下略)
    """
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.scatter(X1, X2, c=labels_color)
    plt.title(title)
    plt.savefig(file_name)
    
def main():
    
    df = load_iris(as_frame=True).frame         # データフレームの読み込み sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm) target
    N = 0.08                                    # 教師データを残す割合(1.0 ~ 0.0)
    del_num = int((1 - N) * df.shape[0])        # ラベルを-1に変換する個数
    
    shuffle_df = shuffle(df, del_num)           # データフレームをシャッフル
    X = shuffle_df.iloc[:,:-1].to_numpy()               # 特徴量を取得
    y = shuffle_df["target"].to_numpy()                 # 正解データを取得
    labels = y.copy()                           # ラベル加工を行う用の変数を生成
    
    X1 = X[:, 0]
    X2 = X[:, 1]
            
    machining_labels = label_machining(del_num, labels)  # (1 - N)%のラベルを削除(-1)とする

    # 元のラベルデータを表示
    y_labels_color = list(map(lambda label : make_label_color(label), y))
    plot(X1, X2, y_labels_color, "exe10_01加工前のラベル.png", "加工前のラベル")

    # 加工したラベルデータ結果の表示
    machining_labels_color = list(map(lambda label : make_label_color(label), machining_labels))
    plot(X1, X2, machining_labels_color, "exe10_01加工後ラベル.png", "加工後のラベル")

    # SVCによる予測
    svc = SVC(gamma="scale")
    svc.fit(X[del_num:, :], machining_labels[del_num:])
    svc_pred = svc.predict(X)    
    svc_pred_color = list(map(lambda label: make_label_color(label), svc_pred))
    plot(X1, X2, svc_pred_color, "exe10_01_svc.png", "SVCによる予測結果")
    
    # ラベル拡散法による予測
    
    # try:
    #     ls = LabelSpreading()
    #     param = {
    #         "alpha": [0.2, 0.4, 0.5],
    #     }
    #     ls = HalvingRandomSearchCV(ls, param)
    #     ls.fit(X, machining_labels)
    # except:
    #     ls =  LabelSpreading(kernel='knn', alpha=0.2, n_neighbors=7,max_iter=1000, n_jobs=-1)
    #     ls.fit(X, machining_labels)
    
    ls =  LabelSpreading(kernel='knn', alpha=0.2, n_neighbors=7,max_iter=1000, n_jobs=-1)
    ls.fit(X, machining_labels)
        
    ls_pred = ls.predict(X)
    ls_pred_color = list(map(lambda label: make_label_color(label), ls_pred))
    plot(X1, X2, ls_pred_color, "exe10_01_ls.png", "LabelSpreadingによる予測結果")

    print(f'SVCの正解率 {accuracy_score(y, svc_pred) * 100}')
    print(f'LabelSpreadingの正解率 {accuracy_score(y, ls_pred) * 100}')

if __name__ == "__main__":
    main()