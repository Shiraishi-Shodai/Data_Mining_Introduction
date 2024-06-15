import time
import numpy as np

data_XY=[
    [ 1.0 , 3.1 ] ,
    [ 2.0 , 5.1 ] ,
    [ 3.0 , 6.8 ] ,
    [ 4.0 , 8.9 ] ,
    [ 5.0 , 11.5 ] ,
    [ 6.0 , 13.8 ] ,
]
#  w_0 = 2.13714285714286, w_1 = 0.720000000000000}

#偏微分する関数
def diff_func_p(f,vec_w,h=0.0001) :
    grad = np.zeros_like(vec_w)
    for i in range(len(vec_w)):
        vec_i_org = vec_w[i]
        vec_w[i] = vec_i_org + h
        fh1=f(vec_w)
        vec_w[i] = vec_i_org - h
        fh2=f(vec_w)
        grad[i] = (fh1-fh2)/(2*h)
        vec_w[i] = vec_i_org
    return grad

#偏微分する関数（解析的に）
def diff_func_p2(f,vec_w,h=0.0001) :
    a = vec_w[0]
    b = vec_w[1]
    return np.array([182.0*a + 42.0*b - 419.2,42.0*a + 12.0*b - 98.4])
    #return grad

#損失関数
def Error_func(vec_w):
    x = vec_w[0]
    y = vec_w[1]
    global data_XY #グローバル変数。イケテないが
    
    E=0
    for li in data_XY:
        E += (li[1]-(x*li[0]+y))**2
    
    return E

eta = 0.002 #学習係数
vec_w=np.array([3.1,1.0]) #重みの初期値

for epoc in range(1,3000):
    grad = diff_func_p(Error_func,vec_w,0.000001)
    vec_w = vec_w - eta * grad
    
    print(str(epoc) +" th train : a=w_1= "+str(vec_w[0]) + ' , b=w_2 = '+str(vec_w[1]),end=" ")
    print("E="+str(Error_func(vec_w)) + " , grad= " +str(grad) )
	
	# 勾配がほぼ0になったら止める
    if (grad[0]*grad[0] +grad[1]*grad[1])**0.5 < 0.00001:
        break