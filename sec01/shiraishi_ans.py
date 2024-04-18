
import numpy as numpy
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('./apple_quality.csv')
df = df.drop(columns=['A_id'])
df_y = df["Quality"] # コピーしておかないと後の処理で消えちゃう
df_x = df.drop("Quality", axis = 1) #これだと消えない。
#df_x = df.drop("Quality", axis = 1,inplace=True) #これだと消える。
#   df = df.drop(columns=['Quality'])
print(df_y)
print(df_x)
print(df_y)

"""
これは失敬。恥ずかしながら、dropはオブジェクトからデータを削除するメソッドではないようです。
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop.html
dropは、指定したラベルを行または列から削除するだけなので、列そのもののデータを削除するわけではありません。
df_y でdfから列そのものを削除しない限りオブジェクトは残っているので、消えませんね。
失礼しました。
調べると、inplace=True でオブジェクトから削除するようです。
インターネットを調べるとdropは列からデータを削除するという説明がなんと多いことか！
嘘情報だらけです。

"""

"""
この図を見ると分かるでしょうか。
縦軸と横軸に正規分布を作り、それをそれぞれプロットすると今回のような散布図になります。
逆に頭の中で縦軸と横軸にまっすぐおろすのです。（「射影」と言います）
縦軸、横軸にそれぞれ射影すると正規分布がなんとなく見えるということです。このあたりは
慣れと一種のセンスですね。センスは努力で磨けますよ。
"""
plt.figure(figsize=(8, 8))
sns.set(style="white")  
df_x['Label'] = df["Quality"].map({'bad':0, 'good':1})
sns.jointplot(x='Size', y='Weight', hue='Label', data=df_x, palette='tab10', s=9)
plt.show()


