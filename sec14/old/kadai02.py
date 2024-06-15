#import time
import numpy as np
#import matplotlib.pyplot as plt
from NyLinearRegression2 import NyLinearRegression2

data_for_train =np.array([
    [ 1.0 , 3.1 ] ,
    [ 2.0 , 5.1 ] ,
    [ 3.0 , 6.8 ] ,
    [ 4.0 , 8.9 ] ,
    [ 5.0 , 11.5 ] ,
    [ 6.0 , 13.8 ] ,
])

model = NyLinearRegression2()
model.train_data = data_for_train
# fit(X,y,eta=0.002,w_init=[1.0,3.0],iteratr=1000)
print(model.fit(data_for_train[:,0],data_for_train[:,1],eta=0.002,iteratr=1000,w_init=[1.0,3.0]))
print(model.predict(np.array([1.0,2.0])))

#fitにかかった時間を返そう
