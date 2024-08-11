from sklearn.semi_supervised import LabelSpreading
import numpy as np
from matplotlib import pyplot as plt
import japanize_matplotlib
import pandas as pd
from sklearn.datasets import load_iris
import math
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

def shuffle(df):
    target_idx = np.array([df.query('@label == target').index for label in df["target"].unique()])
    shuffle_idx = target_idx.T.reshape(-1, 1).flatten()
    shuffle_df = df.reindex(index=shuffle_idx)
    return shuffle_df

def model_comp(X, labels, del_num):
    model1 = SVC()
    model1.fit(X[del_num:], labels[del_num:])
    res1 = model1.predict(X)

    model2 = LabelSpreading(kernel="knn", max_iter=1000, n_jobs=1, alpha=0.8, n_neighbors=7)
    model2.fit(X, labels)
    res2 = model2.transduction_
    
    return res1, res2
    

def main():
    df = load_iris(as_frame=True).frame
    shuffle_df = shuffle(df)
    
    X = shuffle_df.iloc[:, :-1].to_numpy()
    y = shuffle_df["target"].to_numpy()
    supervised_percentage = [0.03, 0.05, 0.08]

    for i in supervised_percentage:
        labels = y.copy()
        del_num = math.floor(len(labels) * (1 - i))
        labels[0:del_num] = np.array([-1] * del_num)
        
        res1, res2 = model_comp(X, labels, del_num)
        print(f'=========教師ありデータの割合: {i * 100} %=========')
        print(f'正解率: {accuracy_score(y, res1) * 100} %')
        print(f'正解率: {accuracy_score(y, res2) * 100} %', end="\n")
    

if __name__ == "__main__":
    main()