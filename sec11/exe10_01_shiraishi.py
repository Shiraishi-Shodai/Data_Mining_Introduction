"""
irisデータを読み込み、ラベルを次の割合でわざと消す。分類アルゴリズムSVMと半教師あり学習（ラベル拡散）を比較してそれぞれの正解率を求め、どちらが正解率が高いか調べよ。

（１）ラベルを全体の3％にしたとき

HalvingRandomSearchCV
初期段階でランダムにパラメータセットをサンプリングし、それぞれを試す。
各ラウンドごとに成績の悪いパラメータセットを切り捨て、残ったセットにより多くのリソースを割り当てまる。
factor : 各ラウンドで次に進むハイパーパラメータセットの数(上位何パーセントのパラメータセットを残すかを指定)
min_resources: 初期ラウンドで各ハイパーパラメータセットを評価する際に使用される最小リソース量
max_resources: 初期ラウンドで各ハイパーパラメータセットを評価する際に使用される最大リソース量

LabelSpreading
transduction_ : フィット中に各項目に割り当てられるラベル。

多クラス分類(クラスが3種類以上の分類): 多クラス分類の場合は評価指標(再現率、適合率、F1スコアなど..)がクラスの数だけ算出される
https://di-acc2.com/analytics/ai/10801/
https://www.chem-station.com/blog/2021/06/ml2.html
マクロ平均: 各クラス毎に評価指標（適合率・再現率など）を計算した後、平均を取る方法
マイクロ平均: 各クラス毎のTP、TN、FP、FNの値を算出した段階で集計し、評価指標（適合率・再現率など）を算出する方法
"""

import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.semi_supervised import LabelSpreading
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.metrics import confusion_matrix

# データの加工をするクラス
class Machine():
    def shuffle(self,df):
        """
        Description: ラベルが012012の順番になるよう並び替え
        
        Args:
            df(pd.DataFrame): load_irisで読み込んだirisデータフレーム
        Return:
            shuffle_df(pd.DataFrame): irisデータフレームを並び替えたデータフレーム
            
        """
        label_types = df["target"].unique()
        # 並び替えたインデックスを取得
        indexes = np.array([df.query("target == @label").index for label in label_types]).reshape(3, -1).T.reshape(-1, 1).flatten()
        # データフレームを並び替え
        shuffle_df = df.reindex(index=indexes)
        
        return shuffle_df
        
    def make_label_color(self, label):
        """
        Description: 整数を受取りそれぞれの値に対応する文字列を返す
        
        Args: 
            label(int) : ラベルデータ(-1~2)
        
        Return:
            文字列(str): black,red, blue, greeのいずれかの文字列を返す
        """
        color_dict = { -1: "black", 0: "red", 1: "blue" }
        return color_dict.get(label, "green")

    def label_machining(self, del_num, labels):
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

class ModelControll():
    
    def __init__(self, model_dict) -> None:
        self.model_dict = model_dict
        self.y_preds = None
        
    def predict(self, X, machining_labels, del_num):
        
        for model_name, model in self.model_dict.items():
        
            match model_name:
                case "SVC":
                    model.fit(X[del_num:, :], machining_labels[del_num:])
                    svc_pred = model.predict(X)
                
                case "KMeans":
                    model.fit(X)
                    km_pred = np.array(list(map(lambda label: 1 if label == 2 else 0 if label == 1 else 2, model.labels_))) # ラベルを揃える           
                    
                case "LabelSpreading":
                    model.fit(X, machining_labels)
                    ls_pred = model.transduction_

        self.y_preds =  [svc_pred, km_pred, ls_pred]
        return self.y_preds        

    def score(self, y):
        
        tmp_score_df = pd.DataFrame(columns=["macro_precision", "macro_recall", "macro_F1", "accuracy"])
        
        for model_name, y_pred in zip(self.model_dict.keys(), self.y_preds):
    
            con_m = confusion_matrix(y, y_pred)
            # 左上、中上、右上、左中、中中、右中、左下、中下、右下
            lu, mu, ru, lm, mm, rm, ll, ml, rl = con_m.flatten()
            label_con_m = {
                'setosa' : {
                    "TN": (mm + rm + ml + rl),
                    "FP": (lm + ll),
                    "FN": (mu + ru),
                    "TP": lu,
                },
            
                'versicolor' : {
                    "TN": (ru + lu + rl + ll),
                    "FP": (mu + ml),
                    "FN": (lm + rm),
                    "TP": mm,
                },
            
                'virginica' : {
                    "TN": (lu + mu + lm + mm),
                    "FP": (rm + rm),
                    "FN": (ll + ml),
                    "TP": rl
                },
            }
            
            # 各ラベルの評価配列
            precision_arr = []
            recall_arr = []
            F1_arr = []
            
            for key, item in label_con_m.items():
                
                if  item["TP"] == 0:
                    precision = 0.0
                    recall = 0.0
                    F1 = 0.0
                
                else:
                    precision = item["TP"] / (item["FP"] + item["TP"]) * 100
                    recall = item["TP"] / (item["FN"] + item["TP"]) * 100     
                    F1 = (2 * precision * recall) / (precision + recall)
                
                precision_arr.append(precision)
                recall_arr.append(recall)
                F1_arr.append(F1)
                # print(f"    ******************{key}の分類結果******************")
                # print(f'    適合率 {precision}%')
                # print(f'    再現率率 {recall}%')
                # print(f"    F1スコア {F1}")
            
            macro_precision = sum(precision_arr) / 3
            macro_recall = sum(recall_arr) / 3
            macro_F1 = sum(F1_arr) / 3
            accuracy = accuracy_score(y, y_pred) * 100
            
            tmp_score_df.loc[model_name] = pd.Series(data=[macro_precision, macro_recall, macro_F1, accuracy],index=["macro_precision", "macro_recall", "macro_F1", "accuracy"])
            
            print(f"⭐⭐⭐⭐⭐⭐⭐⭐　{model_name}の予測結果　⭐⭐⭐⭐⭐⭐⭐⭐")
            print(f'    マクロ適合率  : {macro_precision}')
            print(f'    マクロ再現率  : {macro_recall}')
            print(f'    マクロF1スコア: {macro_F1}')
            print(f'    正解率        : {accuracy}', end='\n')
        
        return tmp_score_df
    
def plot(X1, X2, labels_color, file_name, title, xlabel="sepal width", ylabel="petal length"):
    """グラフを描画する(以下略)
    """
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.scatter(X1, X2, c=labels_color)
    ax.set_title(title)
    fig.savefig(file_name)
    
def main():
    
    print("=============== データの読み込み・前処理 ===============")
    df = load_iris(as_frame=True).frame         # データフレームの読み込み sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm) target
    machine = Machine()
    
    n_list = np.array([0.03, 0.05, 0.08])
    
    # SVCによる予測(教師あり学習)
    svc = SVC(gamma="scale")
    # KMeansによる予測(教師なし学習)
    km = KMeans(n_clusters=3, random_state=0)
    # ラベル拡散法による予測(半教師あり学習)
    ls =  LabelSpreading(kernel='knn', alpha=0.2, n_neighbors=7,max_iter=1000, n_jobs=-1)
    
    model_dict = {
        "SVC": svc,
        "KMeans": km,
        "LabelSpreading": ls
    }
    
    mc = ModelControll(model_dict)
    
    score_columns = ["macro_precision", "macro_recall", "macro_F1", "accuracy"]
    score_df = pd.DataFrame(columns=score_columns)
    
    
    for i, n in enumerate(n_list):
            
        del_num = int((1 - n) * df.shape[0])        # ラベルを-1に変換する個数
        
        shuffle_df = machine.shuffle(df)           # データフレームをシャッフル
        X = shuffle_df.iloc[:,:-1].to_numpy()               # 特徴量を取得
        y = shuffle_df["target"].to_numpy()                 # 正解データを取得
        labels = y.copy()                           # ラベル加工を行う用の変数を生成
        
        X1 = X[:, 1]
        X2 = X[:, 2]
                
        machining_labels = machine.label_machining(del_num, labels)  # (1 - N)%のラベルを削除(-1)とする

        # 元のラベルデータを表示
        y_labels_color = list(map(lambda label : machine.make_label_color(label), y))
        plot(X1, X2, y_labels_color, f"exe10_01_{n}_加工前のラベル.png", f"教師データを{n}%残した時の加工前のラベル")

        # 加工したラベルデータ結果の表示
        machining_labels_color = list(map(lambda label :machine. make_label_color(label), machining_labels))
        plot(X1, X2, machining_labels_color, f"exe10_01_{n}_加工後ラベル.png", f"教師データを{n}%残した時の加工後のラベル")

        print()
        print("=============== 予測 ===============")
    
                
        y_preds = mc.predict(X, machining_labels, del_num)

        for model_name, y_pred in zip(mc.model_dict.keys(), y_preds):
            y_pred_color = list(map(lambda label: machine.make_label_color(label), y_pred))
            plot(X1, X2, y_pred_color, f"exe10_01_{n}_{model_name}_scatter.png", f"教師データを{n}%残した時の{model_name}による予測結果")

        tmp_score_df = mc.score(y)
        
        if score_df.empty:
            score_df = tmp_score_df
        else:
            score_df = pd.concat([score_df, tmp_score_df], axis=0)
    
    for score in score_columns:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title(f'{score}の推移')
        ax.set_xlabel('教師ラベルが存在する割合(%)')  
        ax.set_ylabel(f'{score} Score')  
        for model_name, marker in zip(mc.model_dict.keys(), ["o", "x", "P"]):

            ax.plot(n_list* 100, score_df.loc[model_name][score], label=model_name, marker=marker)
    
        fig.legend()
        fig.savefig(f'{score}.png')
        
if __name__ == "__main__":
    main()