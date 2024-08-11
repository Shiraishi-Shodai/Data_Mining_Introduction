import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

df = pd.read_csv('group_sample.csv')
X = df[["x1", "x2"]].to_numpy()
y = df["y"].to_numpy()
# オリジナルデータのプロット
# plt.scatter(X[:, 0], X[:, 1])
# plt.savefig('original.png')

def count_cluster(labels)-> int:
    if -1 in set(labels):
        return len(list(set(labels))) - 1
    else:
        return len(list(set(labels)))
        
def grid_search(grid_parameters)->tuple[dict, int, list]:
    
    best_parameters = {"min_samples":0, "eps": 0}
    best_score = 0
    pred_labels = []
    
    for min_samples in grid_parameters["min_samples"]:
        for eps in grid_parameters["eps"]:
            model = DBSCAN(min_samples=min_samples, eps=eps)
            model.fit(X)
            
            if count_cluster(model.labels_) == 3:
                encoding_label = list(map(lambda x: x if x == -1 else x + 1, model.labels_))
                score = accuracy_score(y, encoding_label)
                if best_score < score:
                    best_parameters.update(min_samples=min_samples, eps=eps)
                    best_score = score
                    pred_labels = encoding_label
           
    return best_parameters, best_score, pred_labels

grid_parameters = {"min_samples":np.arange(2, 20, 1), "eps": np.arange(0.1, 3, 0.01)}
res = grid_search(grid_parameters)

if res[1] == 0:
    print("3クラスターに分割できませんでした。")
else:
    # 予測データのプロット
    print(res[2])
    plt.scatter(X[:, 0], X[:, 1], c=res[2])
    plt.savefig('pred.png')
    print(f"ベストパラメータ: {res[0]}")
    print(f"ベストスコア: {res[1]}")