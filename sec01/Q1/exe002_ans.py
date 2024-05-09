# 2024年4月11日の第001回の演習問題
#　1234番　山本幸太郎

"""
まず、普段よく使うモジュールをソースコードの冒頭でまとめてロードします。
Visual Studio Codeでは使われていないモジュール色が変わります。
よって不要なモジュールをロードしている箇所は先頭に#を付けてコメントアウトしてしまえばよいです。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import recall_score,precision_score
from sklearn.metrics import roc_curve, auc


"""
データサイエンスではまずはじめにデータをかるくざっと眺めたり、最低限の処理（欠損値の処理など）をします。

"""
#(1)A_id以外の列をdfというデータフレームにする
df = pd.read_csv("apple_quality.csv")
df = df.drop("A_id", axis = 1)
print("----データ分析と前処理-------------------------------------------------")
print("データフレーム概要")
print(df.info())

print("\n")
print("各説明変数概要")
print(df.describe())

#(5)good は 1 に、badは 0 にラベルエンコーディング
print("\n")
print("数値でない列があるので、Qualityをラベルエンコーディングします。")
df["Quality"] = df["Quality"].map({'bad':0, 'good':1})
print("データフレーム概要を再確認")
print(df.info())

print("\n")
print("重複した行がないか確認します。")
print(df.duplicated().sum())

print("\n")
print("欠損値を各特徴量について確認します。")
print(df[df.isnull().any(axis = 1)])

print("----データ分析と前処理-------------------------------------------------")

#(2)Size列,Weight列の最小値、最大値、平均値
size_min=np.min(df['Size'])
size_max=np.max(df['Size'])
size_mean=np.mean(df['Size'])

weight_min=np.min(df['Weight'])
weight_max=np.max(df['Weight'])
weight_mean=np.mean(df['Weight'])
print(f"Sizeの最小値、最大値、平均値は{size_min},{size_max},{size_mean}")
print(f"Weightの最小値、最大値、平均値は{weight_min},{weight_max},{weight_mean}")
# describeではもう少し多くの情報が取れる。
print(df['Size'].describe())
print(df['Weight'].describe())

print("各特徴量の分布をヒストグラムで眺めます。")
numerical_cols = ['Size', 'Weight', 'Sweetness', 'Crunchiness', 'Juiciness', 'Ripeness', 'Acidity',"Quality"]
plt.figure(figsize=(15, 10))
sns.set_palette("tab10")
for i, column in enumerate(numerical_cols, 1):
    plt.subplot(3, 3, i)
    sns.histplot(data=df, x=column, kde=True, bins=20)  # Use sns.histplot for newer versions
    plt.title(column)
plt.tight_layout()
plt.savefig("param_hist.png")
plt.show()

print("各特徴量の相関をヒートマップで眺めます。")
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='Reds', fmt='.2f', linewidths=.5)
plt.savefig("corr_heatmap.png")
plt.show()

print("ここまで分析して、本データについては以下の様に考えます。")
my_opinion="""
データフレームは説明変数は全てフロート型で、目的変数は0/1の整数型になっています。
説明変数は美しい正規分布に見え、目的変数もほぼ50％と偏りはありません。
不均一なデータではないことがわかります。
特徴量は相関係数は最大でも0.3程度なので、全ての特徴量をこのまま学習に使用します。
"""

print(my_opinion)

#(3) df の先頭 1200 行のデータを取り出し、NumPy配列としてX_train,y_train
df_x = df.drop("Quality", axis = 1)
df_y = df["Quality"]
X = df_x.to_numpy()
y = df_y.to_numpy()

ss = StandardScaler()
X = ss.fit_transform(X)
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
#scaler = MinMaxScaler(feature_range=(0, 1))
#X = scaler.fit_transform(X)
#smote = SMOTE(random_state=42)
#X, y = smote.fit_resample(X, y)

X_train = X[0:1200,:]
y_train = y[0:1200]
#print(X_train)
#print(y_train)

#(4) の 1200 行のデータを用い、横軸に Size を、縦軸に Weight をとって、散布図を描きなさい。
print("(4)のグラフです")
x_size = df['Size'].to_numpy()
y_weight = df['Weight'].to_numpy()
plt.scatter(x_size[0:1200],y_weight[0:1200])
plt.xlabel("size")
plt.ylabel("weight")
plt.savefig("size-weight.png")
plt.show()
#plt.close()

#(5) 先頭1200行で学習
print("(5)まずはSVCでハイパーパラメータはデフォルトで学習します。")
X_test = X[1200:,:]
y_test = y[1200:]
model =SVC()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy(with SVC) = {100*accuracy}%")

print("ランダムサーチで最適なハイパーパラメータを見つけます")
param_dist = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto', 0.05,0.1, 1],
}
"""
param_dist = {
    'C': [8.5,9.5,9.8,10,10.2,10.5,11],
    'kernel': ['rbf'],
    'gamma': [0.08,0.09,0.10,0.11,0.12],
}
"""
svc = SVC()
randomized_search = RandomizedSearchCV(svc, param_distributions=param_dist, n_iter=10, cv=5, scoring='accuracy', random_state=0, n_jobs=-1)
randomized_search.fit(X_train, y_train)
best_params = randomized_search.best_params_
print(f"Best Hyperparameters: {best_params}")

best_model = randomized_search.best_estimator_
y_test_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_test_pred)
print(f"Accuracy(with 調整後SVC ) = {100*accuracy}%")

print("モデルの指標を計算します。")
#正解のデータと予測データを与えて再現率、適合率、正解率などを辞書で返す
def get_scores(y_true, y_pred):
    con_m = confusion_matrix(y_true, y_pred)
    #print(con_m)
    tn, fp, fn, tp = con_m.flatten()
    recall = tp/(tp + fn)
    precision =tp/(tp + fp)
    accuracy = (tp + tn)/(tn + fp + fn + tp )
    return {'recall':recall, 'precision':precision , 'accuracy':accuracy }

print(get_scores(y_test, y_test_pred))

print(f"再現率 = {recall_score(y_test, y_test_pred)}")
print(f"適合率 = {precision_score(y_test, y_test_pred)}")
print(f"正解率 = {accuracy_score(y_test, y_test_pred)}")

print("混同行列を表示します。")
conf_matrix = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(10, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='crest', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.savefig("confusion_matrix.png")
plt.show()

print("ROC曲線とそのAUCを計算します。")
y_prob = best_model.decision_function(X_test) # decision_functionはSVCが最終的に判定する確率を出す
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
print(f"AUC={roc_auc}")

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('Recall(True Positive Rate)')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.savefig("roc.png")
plt.show()