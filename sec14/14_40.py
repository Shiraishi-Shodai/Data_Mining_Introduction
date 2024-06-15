import numpy as np
import torch

data_ex =torch.tensor([
    [ 1.0 , 3.1 ] ,
    [ 2.0 , 5.1 ] ,
    [ 3.0 , 6.8 ] ,
    [ 4.0 , 8.9 ] ,
    [ 5.0 , 11.5 ] ,
    [ 6.0 , 13.8 ] ,
])

x = data_ex[:,0].detach()
y = data_ex[:,1].detach()

x_mean = x.mean()
y_mean = y.mean()

s_xy = sum((x - x_mean) * (y - y_mean))
s_x2 = sum((x - x_mean) ** 2)

a=s_xy/s_x2
b= y.mean()-a*x_mean

# print(x)
# print(y)
print(f"x=7の時の予測値は{a*7.0+b}")
print(f'a: {a}')
print(f'b: {b}')
