import numpy as np

data_ex =np.array([
    [ 1.0 , 3.1 ] ,
    [ 2.0 , 5.1 ] ,
    [ 3.0 , 6.8 ] ,
    [ 4.0 , 8.9 ] ,
    [ 5.0 , 11.5 ] ,
    [ 6.0 , 13.8 ] ,
])

x = data_ex[:,0].copy()
y = data_ex[:,1].copy()

###### この後のコードを修正してみよう。
### x.mean()やlen(x)などを複数ヶ所で使用しており、計算量が増える
### for文も必要ない

s_xy =0
s_x2 =0

#以下がs_x2の計算式
for i in range(0,len(x)):
    s_x2 = s_x2+ (x[i]-x.mean())**2
s_x2 = s_x2/len(x)

#以下がs_xyの計算式
for i in range(0,len(x)):
    #print(i)
    s_xy = s_xy+ (x[i]-x.mean())*(y[i]-y.mean())
s_xy = s_xy/len(x)

a=s_xy/s_x2
b= y.mean()-a*x.mean()

print(f"a={a:.4f},b={b:4f}")
print(f"x=7の時の予測値は{a*7.0+b}")
