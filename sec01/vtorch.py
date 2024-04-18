import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('apple_quality.csv')
df = df.drop("A_id", axis=1)
df = df.replace({"Quality": {"good": 0, "bad": 1}})

#  Xを標準化
X = df.iloc[:, :-1].values
ss = StandardScaler()
X = ss.fit_transform(X)
X = torch.tensor(X, dtype=torch.float)
y = torch.tensor(df["Quality"].values, dtype=torch.long)

X_train, X_val, y_train, y_val = train_test_split(X, y, shuffle=False, test_size=0.3)
print(X_train.shape, y_train.shape)

# Datasetの作成(学習データとテストデータをまとめる)
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

# DataLoaderを作成(バッチ単位でデータセットを取り出しやすくする)
batch_size = 100
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 動作確認
# こんな感じでバッチ単位で取り出すことができます。
# イテレータに変換
batch_iterator = iter(train_dataloader)
# 1番目の要素を取り出す
inputs, labels = next(batch_iterator)
print(inputs.size())
print(labels.size())

model = nn.Sequential(
    nn.Linear(7, 100),
    nn.ReLU(),
    nn.Linear(100, 50),
    nn.ReLU(),
    nn.Linear(50, 2)
)


# 損失関数
criterion = nn.CrossEntropyLoss()
# 最適化関数
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# エポック数
num_epochs = 1

dataloaders_dict = {
    "train": train_dataloader,
    "val" : val_dataloader
}


        
