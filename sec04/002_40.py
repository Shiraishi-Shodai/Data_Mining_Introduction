import numpy as np
import matplotlib.pyplot as plt
from skfuzzy.cluster import cmeans
import pandas as pd

df = pd.read_csv('iris.csv',usecols=['sepal_length','sepal_width','petal_length','petal_width'])
X = df.values
def target_to_color(target):
    if type(target) == np.ndarray:
        return (target[0], target[1], target[2])
    else:
        return "rgb"[target]

m = 2.0
c_means = cmeans(X.T, 3, m, 0.003, 10000)
plt.figure()
plt.scatter(X[:,1], X[:,2], c=[target_to_color(t) for t in c_means[1].T])
#plt.savefig('c_means.png')
plt.show()

print(c_means[1])