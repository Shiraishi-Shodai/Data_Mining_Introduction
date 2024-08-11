import sympy
import numpy as np
import time

start_time = time.time()
data_ex=np.array([
    [1.0,3.1],
    [2.0,5.1],
    [3.0,6.8],
    [4.0,8.9],
    [5.0,11.5],
    [6.0,13.8],
])

def cost_func(data_ex,a,b):
    E=0
    for li in data_ex:
        E += (li[1]-(a*li[0]+b))**2
    return E

# ans = cost_func(data_ex, 0.1, 0.1)

a = 0.1
a_min = a
b = 0.1
b_min = b
h = 0.01
E_min = 10000000
E_current = 0
while a<10 :
    while b < 10:
        E_current = cost_func(data_ex,a,b)
        #print(f'{a},{b} ,E_current ={E_current},E_min ={E_min}')
        if E_current <E_min :
            E_min = E_current
            a_min = a
            b_min = b
        b += h
    a += h
    b = 0.1

print(f'a= {a_min},b={b_min},E={E_min}')
print(f"処理時間は{time.time() - start_time}秒でした")